#!/usr/bin/env bash
# Persistent MOSS-Image-SFT INT4 backend (stable-diffusion.cpp sd-server + Metal).
# Keeps the 4-bit DiT, Qwen3.5-9B text encoder and VAE loaded so each request only pays for
# sampling. The MCP server starts this automatically; run it yourself for HTTP/CLI use.
#
#   http://127.0.0.1:18765/sdcpp/v1/img_gen      native async API (txt2img / img2img / inpaint)
#   http://127.0.0.1:18765/sdapi/v1/txt2img      A1111-compatible
#   http://127.0.0.1:18765/v1/images/generations OpenAI-compatible
#
# Env: MOSS_PORT (default 18765), MOSS_HOST (default 127.0.0.1 - keep it local unless you
# know you want the API on your network; it has no authentication).
set -euo pipefail
cd "$(dirname "$0")"

PORT="${MOSS_PORT:-18765}"
HOST="${MOSS_HOST:-127.0.0.1}"

for f in bin/sd-server models/diffusion_models/MOSS-Image-SFT-Q4_0.gguf \
         models/text_encoders/Qwen3.5-9B-Q4_K_M.gguf models/vae/flux2_ae_small.safetensors \
         models/tokenizers/tokenizer_qwen3.5.json; do
  [[ -e "$f" ]] || { echo "error: $f missing - run ./setup.sh first" >&2; exit 1; }
done

# sd-server loads weights on the first request. If a model file is an online-only cloud
# placeholder (Google Drive / iCloud "dataless"), that request first downloads it, which turned
# the first image into a 5-13 minute wait. Keep models/ on local disk; warn if it is not.
for f in models/diffusion_models/MOSS-Image-SFT-Q4_0.gguf models/text_encoders/Qwen3.5-9B-Q4_K_M.gguf \
         models/vae/flux2_ae_small.safetensors models/tokenizers/tokenizer_qwen3.5.json models/loras/*.safetensors; do
  if [[ "$(stat -L -f %Sf "$f" 2>/dev/null)" == *dataless* ]]; then
    echo "warning: $f is online-only (not on local disk); the first image will wait for it to download." >&2
  fi
done

# Never load a second copy of the models: two backends exhaust GPU memory and produce
# blank images.
if curl -s -m 2 "http://127.0.0.1:${PORT}/sdcpp/v1/capabilities" 2>/dev/null | grep -q 'MOSS-Image-SFT'; then
  echo "A MOSS backend is already running on port ${PORT}; nothing to do." >&2
  exit 0
fi
if lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "error: port ${PORT} is used by another program; set MOSS_PORT to a free port." >&2
  exit 1
fi

exec bin/sd-server \
  --diffusion-model models/diffusion_models/MOSS-Image-SFT-Q4_0.gguf \
  --llm models/text_encoders/Qwen3.5-9B-Q4_K_M.gguf \
  --vae models/vae/flux2_ae_small.safetensors \
  --tokenizer models/tokenizers/tokenizer_qwen3.5.json \
  --lora-model-dir models/loras \
  --listen-ip "$HOST" \
  --listen-port "$PORT" \
  "$@"
