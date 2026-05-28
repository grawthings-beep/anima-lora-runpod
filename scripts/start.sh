#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="$(command -v python || command -v python3 || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "ERROR: neither python nor python3 was found in PATH." >&2
  exit 2
fi

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace/anima-lora}"
MODEL_ROOT="${MODEL_ROOT:-${WORKSPACE_DIR}}"
CONFIG_DIR="${CONFIG_DIR:-${WORKSPACE_DIR}/config}"
MODEL_MANIFEST="${MODEL_MANIFEST:-${CONFIG_DIR}/model-manifest.json}"

mkdir -p \
  "${MODEL_ROOT}/models/diffusion_models" \
  "${MODEL_ROOT}/models/text_encoders" \
  "${MODEL_ROOT}/models/vae" \
  "${MODEL_ROOT}/models/loras" \
  "${WORKSPACE_DIR}/datasets/train" \
  "${WORKSPACE_DIR}/outputs" \
  "${WORKSPACE_DIR}/cache" \
  "${CONFIG_DIR}"

export HF_HOME="${HF_HOME:-${WORKSPACE_DIR}/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ -n "${MODEL_MANIFEST_JSON:-}" ]]; then
  printf '%s' "${MODEL_MANIFEST_JSON}" > "${MODEL_MANIFEST}"
elif [[ -n "${MODEL_MANIFEST_URL:-}" ]]; then
  "${PYTHON_BIN}" - "${MODEL_MANIFEST_URL}" "${MODEL_MANIFEST}" <<'PY'
import pathlib
import sys
import urllib.request

url, output = sys.argv[1], pathlib.Path(sys.argv[2])
output.parent.mkdir(parents=True, exist_ok=True)
request = urllib.request.Request(url, headers={"User-Agent": "runpod-anima-lora-template"})
with urllib.request.urlopen(request, timeout=60) as response:
    output.write_bytes(response.read())
PY
elif [[ ! -f "${MODEL_MANIFEST}" && -f /opt/runpod-anima-lora/config/model-manifest.json ]]; then
  cp /opt/runpod-anima-lora/config/model-manifest.json "${MODEL_MANIFEST}"
fi

if [[ "${DOWNLOAD_MODELS:-1}" == "1" && -f "${MODEL_MANIFEST}" ]]; then
  "${PYTHON_BIN}" /opt/runpod-anima-lora/scripts/download_models.py \
    --manifest "${MODEL_MANIFEST}" \
    --root "${MODEL_ROOT}"
else
  echo "Skipping model downloads."
fi

if [[ "${RUN_DEP_CHECK:-1}" == "1" ]]; then
  "${PYTHON_BIN}" /opt/runpod-anima-lora/scripts/check_env.py --workspace "${WORKSPACE_DIR}"
fi

if [[ "${AUTO_TRAIN:-0}" == "1" ]]; then
  exec /opt/runpod-anima-lora/scripts/train_lora.sh
fi

cat <<EOF
Anima LoRA trainer (sd-scripts) is ready.

Dataset (images + matching .txt captions):
  ${WORKSPACE_DIR}/datasets/train

Model root:
  ${MODEL_ROOT}/models

Commands:
  # (optional) generate dataset.toml manually; train_lora.sh does this automatically
  /opt/runpod-anima-lora/scripts/make_metadata.py --dataset-dir ${WORKSPACE_DIR}/datasets/train
  /opt/runpod-anima-lora/scripts/train_lora.sh

Output:
  ${WORKSPACE_DIR}/outputs/${OUTPUT_NAME:-anima_lora}.safetensors
EOF

sleep infinity
