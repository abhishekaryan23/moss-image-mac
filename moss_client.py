#!/usr/bin/env python3
"""Client for the local MOSS-Image server (stable-diffusion.cpp sd-server).

Start the server first:  ./start_server.sh
Then, from Python:
    from moss_client import (
        generate, inpaint, layout_prompt,
        cutout_prompt, extract_cutout_rgba, generate_cutout,
    )
    generate("a red fox in snow", width=768, height=512, out="fox.png")
    generate_cutout("a vintage brass compass", width=768, height=768, out="compass.png")

Or from the shell:
    python3 moss_client.py "a red fox in snow" --width 768 --height 512 -o fox.png
    python3 moss_client.py "a red scarf" --init photo.png --mask mask.png -o edited.png
    python3 moss_client.py --layout layout.json --width 768 --height 768 -o poster.png
    python3 moss_client.py "a porcelain teacup" --cutout --bg-color white -o teacup.png
"""
import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_PORT = 18765  # not 7860: that is Stable Diffusion WebUI's default and often taken
SERVER = os.environ.get("MOSS_SERVER", f"http://127.0.0.1:{os.environ.get('MOSS_PORT', DEFAULT_PORT)}")
MODEL_NAME = "MOSS-Image-SFT-Q4_0.gguf"  # used to verify the server on SERVER is ours

# Canvas limits. The turbo LoRAs are trained at <=1024^2; larger canvases also risk running
# out of GPU memory on 16 GB Macs. Area cap ~1.1 MP keeps every preset and 1024x1024 legal.
MIN_SIDE, MAX_SIDE, MAX_PIXELS = 256, 1536, 1024 * 1024 + 64 * 1024
POLL_INTERVAL = 0.5  # seconds between job-status polls

# Turbo (distilled) LoRA: 4 steps, no CFG. The noise levels below are the release
# MOSSImageScheduler (shift 5) snapped grid, i.e. the anchors the LoRA was trained on.
# Each LoRA is trained for exactly its own step count, so it is paired with its sigmas.
TURBO = {
    3: ("moss_3step_f16.safetensors", [1.0, 0.835, 0.024, 0.0]),       # draft / fastest
    4: ("moss_4step_f16.safetensors", [1.0, 0.91, 0.717, 0.024, 0.0]),  # standard
}

# Any other step count runs the undistilled model with the official settings
# (no LoRA, CFG 3, release scheduler with shift 5, negative branch skips DiT layer 22).
# Same memory as turbo, ~25x slower: the reference "high" quality.
HIGH_STEPS = 50
HIGH_CFG = 3.0
QUALITY_STEPS = {"draft": 3, "standard": 4, "best": 4, "high": HIGH_STEPS}

# Solid studio background presets for single-object cutout generation.
BG_COLORS = {
    "white": ("#FFFFFF", (255, 255, 255), "pure flat solid #FFFFFF studio white background"),
    "green": ("#00FF00", (0, 255, 0), "pure flat solid #00FF00 chroma key green background"),
    "magenta": ("#FF00FF", (255, 0, 255), "pure flat solid #FF00FF chroma key magenta background"),
    "gray": ("#808080", (128, 128, 128), "pure flat solid #808080 neutral 50% gray background"),
    "grey": ("#808080", (128, 128, 128), "pure flat solid #808080 neutral 50% gray background"),
    "black": ("#000000", (0, 0, 0), "pure flat solid #000000 studio black background"),
    "blue": ("#0000FF", (0, 0, 255), "pure flat solid #0000FF chroma key blue background"),
}


class MossError(RuntimeError):
    """A generation failed in a way the caller should report as-is (message is user-facing)."""


class BackendUnavailable(MossError):
    """No MOSS backend is reachable (not started, crashed, or restarted mid-job)."""


def _b64(path):
    path = os.path.expanduser(str(path))
    if not os.path.isfile(path):
        raise MossError(f"image not found: {path}")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _request(path, body=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(SERVER + path, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:300]
        raise MossError(f"backend rejected {path} (HTTP {e.code}): {detail}") from None
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
        raise BackendUnavailable(f"MOSS backend not reachable at {SERVER} ({e})") from None


def _post(path, body):
    return _request(path, body)


def _get(path):
    return _request(path)


def server_identity():
    """Return the backend's capabilities if SERVER is a MOSS sd-server, None if nothing listens.

    Raises MossError if something else (e.g. another web UI) is listening on that port.
    """
    try:
        caps = _request("/sdcpp/v1/capabilities", timeout=3)
    except BackendUnavailable:
        return None
    except MossError:
        caps = None
    name = ((caps or {}).get("model") or {}).get("name", "")
    if name != MODEL_NAME:
        raise MossError(f"port {SERVER} is used by another service (model={name or 'unknown'}). "
                        "Set MOSS_PORT to a free port.")
    return caps


def _snap16(v):
    return max(16, int(round(v / 16)) * 16)


def validate_size(width, height):
    """Snap to multiples of 16 and reject canvases that are too small/large for this model."""
    w, h = _snap16(width), _snap16(height)
    if min(w, h) < MIN_SIDE or max(w, h) > MAX_SIDE or w * h > MAX_PIXELS:
        raise MossError(f"canvas {w}x{h} out of range: each side {MIN_SIDE}-{MAX_SIDE} px and at most "
                        f"~{MAX_PIXELS / 1e6:.1f} MP (e.g. 1024x1024, 1344x768). Larger sizes lose "
                        "quality with the turbo LoRA and can exhaust GPU memory.")
    return w, h


def layout_prompt(caption, items, width, height):
    """Build a MOSS-Image layout (bounding-box) prompt.

    items: list of dicts {"label": str, "caption": str, "box": (x0, y0, x1, y1)}
           with box in normal top-left pixel coords of the output canvas.
           Converted to the model's native yxyx, centre-origin coordinates.
    """
    cx, cy = width / 2, height / 2
    out = []
    for it in items:
        box = it.get("box")
        if not box or len(box) != 4:
            raise MossError("each layout item needs box: [x0, y0, x1, y1] in output pixels")
        x0, y0, x1, y1 = (float(v) for v in box)
        x0, x1 = sorted((max(0.0, min(width, x0)), max(0.0, min(width, x1))))
        y0, y1 = sorted((max(0.0, min(height, y0)), max(0.0, min(height, y1))))
        if x1 - x0 < 8 or y1 - y0 < 8:
            raise MossError(f"layout box {box} is empty or outside the {width}x{height} canvas")
        out.append({
            "label": it.get("label", "object"),
            "caption": it["caption"],
            "bbox": [round(y0 - cy), round(x0 - cx), round(y1 - cy), round(x1 - cx)],
        })
    layout = {"summary": {"caption": caption}, "items": out,
              "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
    return json.dumps(layout, ensure_ascii=False)


def _cancel(job_id):
    try:
        _post(f"/sdcpp/v1/jobs/{job_id}/cancel", {})
    except MossError:
        pass


def _looks_blank(png_bytes):
    """True if the image is a near-uniform frame: what a Metal out-of-memory failure returns."""
    import io
    from PIL import Image, ImageStat
    img = Image.open(io.BytesIO(png_bytes)).convert("L").resize((64, 64))
    return ImageStat.Stat(img).stddev[0] < 2.0


def _run(body, out, timeout=600, on_progress=None, cancel_event=None):
    """Submit a job, report progress, and write the PNG to `out`.

    on_progress(step, total, message) is called as the job moves through the queue and sampler.
    cancel_event (threading.Event) cancels the server job when set. Any failure, timeout or
    interruption cancels the job so it never blocks the queue for other callers.
    """
    if on_progress and not body.get("lora"):
        # Step-level progress needs cheap latent-projection previews (~1% overhead), so they are
        # only enabled for long full-model ("high") renders; turbo jobs report queue/run state.
        body = {**body, "preview": "proj", "preview_interval": 1}
    job = _post("/sdcpp/v1/img_gen", body)
    job_id = job["id"]
    t0 = time.time()
    done = False
    try:
        while True:
            st = _get(f"/sdcpp/v1/jobs/{job_id}")
            status = st.get("status")
            if status == "completed":
                break
            if status in ("failed", "cancelled"):
                err = st.get("error") or {}
                msg = err.get("message") if isinstance(err, dict) else err
                raise MossError(f"generation {status}: {msg or 'no details from backend'}")
            if on_progress:
                if status == "queued":
                    on_progress(0, None, f"queued (position {st.get('queue_position', '?')})")
                else:
                    pv = st.get("preview") or {}
                    if pv.get("total_steps"):
                        on_progress(pv["step"], pv["total_steps"],
                                    f"sampling step {pv['step']}/{pv['total_steps']}")
                    else:
                        on_progress(0, None, "encoding prompt / loading model")
            if cancel_event is not None and cancel_event.is_set():
                raise MossError("cancelled")
            if time.time() - t0 > timeout:
                raise MossError(f"timed out after {timeout:.0f} s")
            time.sleep(POLL_INTERVAL)
        done = True
    finally:
        if not done:
            _cancel(job_id)
    imgs = (st.get("result") or {}).get("images") or []
    if not imgs:
        raise MossError("backend returned no image")
    data = imgs[0].get("b64_json", "") if isinstance(imgs[0], dict) else imgs[0]
    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    png = base64.b64decode(data)
    if _looks_blank(png) and not os.environ.get("MOSS_ALLOW_FLAT"):
        raise MossError("the backend returned a blank image, which usually means the GPU ran out of "
                        "memory (another model or a second MOSS backend may be loaded). Close other "
                        "GPU-heavy apps and retry. Set MOSS_ALLOW_FLAT=1 if a flat image is intended.")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "wb") as f:
        f.write(png)
    return out, time.time() - t0


def generate(prompt, width=768, height=768, seed=-1, out="out.png", steps=4,
             init_image=None, mask_image=None, strength=0.75, on_progress=None,
             cancel_event=None, ref_images=None):
    """Text-to-image, img2img (init_image) or inpainting (init_image + mask_image).

    ref_images: paths of reference images (the FLUX.2-style multi-reference input that
    sd-server takes as "ref_images"): the model keeps their subjects' look, e.g. one
    character drawn in a new pose. They combine with text-to-image, img2img or inpainting.

    steps: 3 (draft) or 4 (standard) use the turbo LoRAs; any other value (e.g. 50 = "high")
    runs the full model with the official CFG 3 settings. Mask: white = repaint, black = keep.
    Sizes are snapped to multiples of 16 and validated. on_progress(step, total, message) is
    called while the job runs; setting cancel_event cancels it.
    """
    if not str(prompt or "").strip():
        raise MossError("prompt is empty")
    width, height = validate_size(width, height)
    if not 0.0 < float(strength) <= 1.0:
        raise MossError("strength must be in (0, 1]")
    if steps in TURBO:
        lora, sigmas = TURBO[steps]
        sample = {"sample_method": "euler", "sample_steps": len(sigmas) - 1,
                  "custom_sigmas": sigmas, "guidance": {"txt_cfg": 1.0}}
        loras = [{"path": lora, "multiplier": 1.0, "is_high_noise": False}]
        timeout = 600
    else:
        if not 1 <= int(steps) <= 100:
            raise MossError("steps must be 3, 4 (turbo) or 1-100 for the full model")
        sample = {"sample_method": "euler", "sample_steps": int(steps),
                  "guidance": {"txt_cfg": HIGH_CFG}}
        loras = []
        timeout = max(600, int(steps) * 2 * 60)
    body = {
        "prompt": prompt,
        "negative_prompt": "",
        "width": width,
        "height": height,
        "seed": int(seed),
        "strength": float(strength),
        "batch_count": 1,
        "sample_params": sample,
        "lora": loras,
        "output_format": "png",
    }
    if init_image:
        body["init_image"] = _b64(init_image)
    if mask_image:
        body["mask_image"] = _b64(mask_image)
    if ref_images:
        body["ref_images"] = [_b64(r) for r in ref_images]
    return _run(body, out, timeout=timeout, on_progress=on_progress, cancel_event=cancel_event)


def upscale(path, out=None, scale=2):
    """Lanczos resize. (RealESRGAN on this M4 took ~80 s per image, so it isn't used.)"""
    from PIL import Image
    t0 = time.time()
    img = Image.open(path)
    img = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
    out = out or path
    img.save(out)
    return out, time.time() - t0


def inpaint(prompt, init_image, mask_image, out="inpaint.png", strength=1.0, **kw):
    from PIL import Image
    w, h = Image.open(init_image).size
    return generate(prompt, width=w, height=h, init_image=init_image,
                    mask_image=mask_image, strength=strength, out=out, **kw)


def _resolve_bg_color(bg_color: str = "white") -> tuple[str, tuple[int, int, int], str]:
    """Normalize a background color name or #RRGGBB hex string into (hex, rgb_tuple, description)."""
    key = (bg_color or "white").strip().lower()
    if key in BG_COLORS:
        return BG_COLORS[key]
    raw_hex = key.lstrip("#")
    if len(raw_hex) == 6 and all(c in "0123456789abcdef" for c in raw_hex):
        rgb = (int(raw_hex[0:2], 16), int(raw_hex[2:4], 16), int(raw_hex[4:6], 16))
        hex_upper = f"#{raw_hex.upper()}"
        return hex_upper, rgb, f"pure flat solid {hex_upper} background"
    raise ValueError(
        f"Unsupported bg_color '{bg_color}'. Use one of {sorted(BG_COLORS)} or a #RRGGBB hex code."
    )


def _sample_border_bg_color(rgb_img) -> tuple[int, int, int]:
    """Estimate the solid background RGB color from the median of the 1px outer perimeter."""
    w, h = rgb_img.size
    px = rgb_img.load()
    rs, gs, bs = [], [], []
    for x in range(w):
        for y in (0, h - 1) if h > 1 else (0,):
            r, g, b = px[x, y]
            rs.append(r)
            gs.append(g)
            bs.append(b)
    for y in range(1, h - 1):
        for x in (0, w - 1) if w > 1 else (0,):
            r, g, b = px[x, y]
            rs.append(r)
            gs.append(g)
            bs.append(b)
    rs.sort()
    gs.sort()
    bs.sort()
    mid = len(rs) // 2
    return rs[mid], gs[mid], bs[mid]


def cutout_prompt(object_description: str, width: int = 768, height: int = 768,
                  bg_color: str = "magenta", use_layout_box: bool = True,
                  margin: float = 0.10) -> str:
    """Build a prompt optimized for generating a single isolated object on a flat solid backdrop.

    Defaults to bg_color="magenta" (#FF00FF chroma key) for effortless exterior + interior-hole
    cutout extraction. When use_layout_box=True, wraps the object in MOSS-Image's native
    abs_centralized yxyx bounding-box layout with a padded margin (default 10% on each side)
    so the outer border is guaranteed object-free solid background and the subject never clips.
    """
    desc = (object_description or "").strip()
    if not desc:
        raise ValueError("object_description must not be empty")
    if not (0.0 <= margin < 0.45):
        raise ValueError("margin must be between 0.0 and 0.45")

    width, height = _snap16(width), _snap16(height)
    hex_code, _, bg_desc = _resolve_bg_color(bg_color)

    isolation_rules = (
        f"Isolated on a {bg_desc} ({hex_code}), completely uniform flat matte chroma-key backdrop, "
        "zero cast shadows, no floor shadow, no contact shadow, no ground plane, "
        "no horizon line, no vignette, no studio gradient, crisp razor-sharp outer silhouette edges, "
        "even neutral studio lighting with zero background color spill on the object, no text, no watermark."
    )

    if not use_layout_box:
        return (
            f"Single isolated object centered in full view: {desc}. "
            f"Full object completely inside frame with generous empty margin around all sides. "
            f"{isolation_rules}"
        )

    x0 = max(16, int(round(width * margin)))
    y0 = max(16, int(round(height * margin)))
    x1 = min(width - 16, int(round(width * (1.0 - margin))))
    y1 = min(height - 16, int(round(height * (1.0 - margin))))

    scene_caption = (
        f"Studio product isolation shot on a {bg_desc}. "
        "Completely uniform flat solid color backdrop filling the entire frame from edge to edge "
        "and visible through any open rings or holes, zero cast shadows, no floor, no horizon line, "
        "no gradient, no vignette, no text."
    )
    item_caption = (
        f"Single complete isolated object centered inside box: {desc}, "
        f"floating against a {bg_desc}, zero cast shadow, crisp clean silhouette edges."
    )
    return layout_prompt(
        scene_caption,
        [{"label": "object", "caption": item_caption, "box": (x0, y0, x1, y1)}],
        width,
        height,
    )


def extract_cutout_rgba(image_path: str, out_path: str | None = None,
                        tolerance: int = 28, feather: int = 2,
                        bg_color: tuple[int, int, int] | str | None = None,
                        remove_holes: bool | None = None) -> str:
    """Extract a clean RGBA cutout from a solid-background image (chroma-key or neutral studio).

    1. Samples the outer perimeter median RGB color (or uses an explicit bg_color).
    2. Runs a border-seeded flood-fill over pixels within `tolerance` of the background color.
    3. Enclosed hole removal (`remove_holes`):
       - Defaults to True for saturated chroma-key backgrounds (magenta #FF00FF, green #00FF00) so
         open rings, handles, and loops are automatically cut out while interior white highlights
         stay 100% intact.
       - Defaults to False on neutral/white backgrounds to preserve large interior white features,
         or can be explicitly set to True to remove enclosed white holes.
    4. Applies a continuous alpha ramp, foreground color unmixing, and luminance-preserving
       broadcast chroma spill suppression along both outer and inner hole boundaries.
    """
    from PIL import Image, ImageChops, ImageDraw, ImageFilter

    if tolerance < 0 or tolerance > 255:
        raise ValueError("tolerance must be between 0 and 255")

    src = Image.open(image_path).convert("RGBA")
    rgb = src.convert("RGB")
    w, h = rgb.size

    if bg_color is None:
        bg_r, bg_g, bg_b = _sample_border_bg_color(rgb)
    elif isinstance(bg_color, str):
        _, (bg_r, bg_g, bg_b), _ = _resolve_bg_color(bg_color)
    else:
        bg_r, bg_g, bg_b = int(bg_color[0]), int(bg_color[1]), int(bg_color[2])

    is_magenta_key = (bg_r > 180 and bg_b > 180 and bg_g < 100)
    is_green_key = (bg_g > 180 and bg_r < 100 and bg_b < 100)
    is_chroma_key = is_magenta_key or is_green_key or (max(bg_r, bg_g, bg_b) - min(bg_r, bg_g, bg_b) >= 120)
    if remove_holes is None:
        remove_holes = is_chroma_key

    # Compute per-pixel Chebyshev color distance max(|R-bg_r|, |G-bg_g|, |B-bg_b|) in C
    diff = ImageChops.difference(rgb, Image.new("RGB", (w, h), (bg_r, bg_g, bg_b)))
    dr, dg, db = diff.split()
    dist_map = ImageChops.lighter(ImageChops.lighter(dr, dg), db)

    eff_tol = max(tolerance, 48) if is_chroma_key else tolerance
    cand = dist_map.point(lambda d: 255 if d <= eff_tol else 0)

    # Also catch desaturated/shaded magenta or green chroma pixels inside small apertures
    if is_magenta_key or is_green_key:
        rgb_px = rgb.load()
        cand_px = cand.load()
        dist_px_init = dist_map.load()
        for y in range(h):
            for x in range(w):
                r, g, b = rgb_px[x, y]
                if is_magenta_key:
                    dom = min(r, b) - g
                    if dom >= 95 and g <= 115 and abs(r - b) <= 65:
                        cand_px[x, y] = 255
                        dist_px_init[x, y] = min(dist_px_init[x, y], max(0, eff_tol - dom // 3))
                elif is_green_key:
                    dom = g - max(r, b)
                    if dom >= 95 and max(r, b) <= 115 and abs(r - b) <= 65:
                        cand_px[x, y] = 255
                        dist_px_init[x, y] = min(dist_px_init[x, y], max(0, eff_tol - dom // 3))

    # Flood-fill exterior background starting from perimeter pixels (marks background as 128)
    for x in range(w):
        if cand.getpixel((x, 0)) == 255:
            ImageDraw.floodfill(cand, (x, 0), 128, thresh=0)
        if h > 1 and cand.getpixel((x, h - 1)) == 255:
            ImageDraw.floodfill(cand, (x, h - 1), 128, thresh=0)
    for y in range(1, h - 1):
        if cand.getpixel((0, y)) == 255:
            ImageDraw.floodfill(cand, (0, y), 128, thresh=0)
        if w > 1 and cand.getpixel((w - 1, y)) == 255:
            ImageDraw.floodfill(cand, (w - 1, y), 128, thresh=0)

    # Detect enclosed background holes (e.g. inside rings/handles) that couldn't be reached from border
    if remove_holes:
        c_px = cand.load()
        d_px = dist_map.load()
        visited = bytearray(w * h)
        for y in range(1, h - 1):
            row_off = y * w
            for x in range(1, w - 1):
                if c_px[x, y] == 255 and not visited[row_off + x]:
                    q = [(x, y)]
                    visited[row_off + x] = 1
                    sum_d = 0
                    min_d = 255
                    for cx, cy in q:
                        d_val = d_px[cx, cy]
                        sum_d += d_val
                        if d_val < min_d:
                            min_d = d_val
                        for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                            if 0 <= nx < w and 0 <= ny < h:
                                n_idx = ny * w + nx
                                if not visited[n_idx] and c_px[nx, ny] == 255:
                                    visited[n_idx] = 1
                                    q.append((nx, ny))
                    area = len(q)
                    avg_d = sum_d / float(area)
                    # Chroma keys: any enclosed hole >= 12px; Neutral bg: uniform flat hole >= 64px with avg_d <= 8, min_d <= 2, and not a tiny artificial test patch
                    is_hole = (area >= 12) if is_chroma_key else (area >= 400 and avg_d <= 6.5 and min_d <= 2)
                    if is_hole:
                        for cx, cy in q:
                            c_px[cx, cy] = 128

    # Background mask: 255 on all confirmed background (exterior + enclosed holes), 0 on object
    ext_bg = cand.point(lambda v: 255 if v == 128 else 0)
    alpha = cand.point(lambda v: 0 if v == 128 else 255)

    # Isolate the 2px boundary ring (for soft alpha) and 7px rim zone (for bounce-light despill)
    dilated_bg = ext_bg.filter(ImageFilter.MaxFilter(5))
    edge_band = ImageChops.subtract(dilated_bg, ext_bg)
    rim_band = ImageChops.subtract(ext_bg.filter(ImageFilter.MaxFilter(15)), ext_bg) if is_chroma_key else edge_band

    soft_span = max(18, int(round(eff_tol * 1.25)))
    soft_upper = min(255, eff_tol + soft_span)

    rgba_px = src.load()
    alpha_px = alpha.load()
    dist_px = dist_map.load()
    edge_px = edge_band.load()
    rim_px = rim_band.load()

    for y in range(h):
        for x in range(w):
            if alpha_px[x, y] == 0:
                rgba_px[x, y] = (0, 0, 0, 0)
                continue
            r, g, b, a_val = rgba_px[x, y]
            if edge_px[x, y] > 0:
                d = dist_px[x, y]
                if d < soft_upper:
                    a_norm = max(0.05, min(1.0, (d - eff_tol * 0.65) / float(max(1, soft_span))))
                    denom = max(a_norm, 0.22)
                    r = min(255, max(0, int(round((r - (1.0 - a_norm) * bg_r) / denom))))
                    g = min(255, max(0, int(round((g - (1.0 - a_norm) * bg_g) / denom))))
                    b = min(255, max(0, int(round((b - (1.0 - a_norm) * bg_b) / denom))))
                    a_val = int(round(a_norm * 255))
                else:
                    a_val = 255
            # Luminance-preserving broadcast chroma despill across edge + rim bounce zone
            if is_magenta_key and rim_px[x, y] > 0:
                rg_ex, bg_ex = r - g, b - g
                if rg_ex > 0 and bg_ex > 0 and min(rg_ex, bg_ex) * 2 >= max(rg_ex, bg_ex):
                    spill = min(rg_ex, bg_ex)
                    r = max(0, r - int(round(spill * 0.65)))
                    b = max(0, b - spill)
                    g = min(255, g + int(round(spill * 0.45)))
            elif is_green_key and rim_px[x, y] > 0:
                ref = (r + b) // 2 if edge_px[x, y] > 0 else max(r, b)
                spill = max(0, g - ref)
                if spill > 0:
                    g = max(0, g - spill)
                    r = min(255, r + int(round(spill * 0.45)))
                    b = min(255, b + int(round(spill * 0.35)))
            alpha_px[x, y] = a_val
            rgba_px[x, y] = (r, g, b, a_val)

    if feather > 0:
        blurred_alpha = alpha.filter(ImageFilter.GaussianBlur(radius=feather * 0.55))
        blur_px = blurred_alpha.load()
        for y in range(h):
            for x in range(w):
                if edge_px[x, y] > 0:
                    r, g, b, _ = rgba_px[x, y]
                    a_val = min(alpha_px[x, y], blur_px[x, y])
                    rgba_px[x, y] = (r, g, b, a_val) if a_val > 0 else (0, 0, 0, 0)

    if out_path is None:
        stem, ext = os.path.splitext(image_path)
        out_path = f"{stem}_rgba{ext or '.png'}"
    src.save(out_path, format="PNG")
    return out_path


def generate_cutout(object_prompt: str, width: int = 768, height: int = 768,
                    seed: int = -1, out: str = "cutout.png", steps: int = 4,
                    bg_color: str = "magenta", transparent: bool = True,
                    use_layout_box: bool = True, tolerance: int = 28,
                    margin: float = 0.10, feather: int = 2,
                    on_progress=None, cancel_event=None) -> tuple[str, str | None, float]:
    """Generate a single isolated object on a solid background and optionally extract an RGBA cutout.

    Returns (plain_path, rgba_path_or_none, total_seconds).
    """
    prompt = cutout_prompt(
        object_prompt,
        width=width,
        height=height,
        bg_color=bg_color,
        use_layout_box=use_layout_box,
        margin=margin,
    )
    plain_path, secs = generate(prompt, width=width, height=height, seed=seed, out=out, steps=steps,
                                on_progress=on_progress, cancel_event=cancel_event)
    rgba_path = None
    if transparent:
        t_rgba = time.time()
        stem, ext = os.path.splitext(plain_path)
        rgba_out = f"{stem}_rgba{ext or '.png'}"
        rgba_path = extract_cutout_rgba(
            plain_path,
            out_path=rgba_out,
            tolerance=tolerance,
            feather=feather,
        )
        secs += time.time() - t_rgba
    return plain_path, rgba_path, secs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("prompt", nargs="?", default="")
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--quality", choices=["draft", "standard", "best", "high"], default=None,
                   help="draft = 3-step turbo, standard = 4-step turbo (default), "
                        "high = official 50 steps / CFG 3 (~10-15 min on a base M4)")
    p.add_argument("--steps", type=int, default=4,
                   help="3 or 4 = turbo LoRA; any other value runs the full model at CFG 3")
    p.add_argument("--upscale", action="store_true", help="2x Lanczos upscale after generation")
    p.add_argument("--init", help="init image for img2img / inpainting")
    p.add_argument("--mask", help="mask image (white = repaint)")
    p.add_argument("--ref", action="append", default=None, help="reference image to keep a subject's look (repeatable)")
    p.add_argument("--strength", type=float, default=None)
    p.add_argument("--layout", help="layout JSON file (MOSS native format, used as prompt)")
    p.add_argument("--cutout", action="store_true",
                   help="generate a single isolated object on a plain background + clean RGBA cutout")
    p.add_argument("--bg-color", default="white",
                   help="solid background color for --cutout (white, green, magenta, gray, black, or #RRGGBB)")
    p.add_argument("--no-transparent", action="store_true",
                   help="with --cutout, skip RGBA background removal and keep only the plain-background PNG")
    p.add_argument("--no-layout-box", action="store_true",
                   help="with --cutout, use plain isolation prompt instead of padded central layout box")
    p.add_argument("--tolerance", type=int, default=28,
                   help="color tolerance (0-255) for border flood-fill RGBA cutout extraction")
    p.add_argument("-o", "--out", default=None)
    a = p.parse_args()
    if a.quality:
        a.steps = QUALITY_STEPS[a.quality]

    prompt = a.prompt
    if a.layout:
        with open(a.layout) as f:
            prompt = json.dumps(json.load(f), ensure_ascii=False)
    if not prompt:
        p.error("prompt or --layout required")
    os.makedirs("outputs", exist_ok=True)
    def cb(step, total, message):
        print(f"\r  {message:<48}", end="", file=sys.stderr, flush=True)
    tag = "cutout" if a.cutout else "moss"
    out = a.out or f"outputs/{tag}_{time.strftime('%Y%m%d_%H%M%S')}.png"
    width, height = a.width, a.height
    if a.init:
        from PIL import Image
        width, height = Image.open(a.init).size
    if a.cutout:
        plain_path, rgba_path, secs = generate_cutout(
            prompt,
            width=width,
            height=height,
            seed=a.seed,
            out=out,
            steps=a.steps,
            bg_color=a.bg_color,
            transparent=not a.no_transparent,
            use_layout_box=not a.no_layout_box,
            tolerance=a.tolerance,
            on_progress=cb,
        )
        if a.upscale:
            _, up = upscale(plain_path)
            secs += up
            if rgba_path:
                _, up_rgba = upscale(rgba_path)
                secs += up_rgba
        if rgba_path:
            print(f"Saved plain {plain_path} and RGBA cutout {rgba_path} in {secs:.1f}s")
        else:
            print(f"Saved plain cutout {plain_path} in {secs:.1f}s")
        return 0

    strength = a.strength if a.strength is not None else (1.0 if a.mask else 0.75)
    path, secs = generate(prompt, width, height, a.seed, out, a.steps, a.init, a.mask, strength,
                          on_progress=cb, ref_images=a.ref)
    if cb:
        print(file=sys.stderr)
    if a.upscale:
        _, up = upscale(path)
        secs += up
    print(f"Saved {path} in {secs:.1f}s")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BackendUnavailable as e:
        sys.exit(f"error: {e}\nStart the backend with ./start_server.sh (or use the MCP server, "
                 "which starts it automatically).")
    except MossError as e:
        sys.exit(f"error: {e}")
    except KeyboardInterrupt:
        sys.exit(130)
