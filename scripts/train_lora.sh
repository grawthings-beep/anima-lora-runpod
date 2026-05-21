#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="$(command -v python || command -v python3 || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "ERROR: neither python nor python3 was found in PATH." >&2
  exit 2
fi

DIFFSYNTH_HOME="${DIFFSYNTH_HOME:-/opt/DiffSynth-Studio}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace/anima-lora}"
MODEL_ROOT="${MODEL_ROOT:-${WORKSPACE_DIR}}"
DATASET_DIR="${DATASET_DIR:-${WORKSPACE_DIR}/datasets/train}"
METADATA_PATH="${METADATA_PATH:-${DATASET_DIR}/metadata.csv}"
OUTPUT_PATH="${OUTPUT_PATH:-${WORKSPACE_DIR}/outputs/anima_lora}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-${MODEL_ROOT}/models/diffusion_models/wai_anima_2859702.safetensors}"
TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:-${MODEL_ROOT}/models/text_encoders/qwen_3_06b_base.safetensors}"
VAE_PATH="${VAE_PATH:-${MODEL_ROOT}/models/vae/qwen_image_vae.safetensors}"
MODEL_PATHS_JSON="${MODEL_PATHS_JSON:-[\"${CHECKPOINT_PATH}\",\"${TEXT_ENCODER_PATH}\",\"${VAE_PATH}\"]}"

TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen/Qwen3-0.6B:./}"
TOKENIZER_T5XXL_PATH="${TOKENIZER_T5XXL_PATH:-stabilityai/stable-diffusion-3.5-large:tokenizer_3/}"
DATA_FILE_KEYS="${DATA_FILE_KEYS:-image}"
MAX_PIXELS="${MAX_PIXELS:-1048576}"
DATASET_REPEAT="${DATASET_REPEAT:-50}"
DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
SAVE_STEPS="${SAVE_STEPS:-500}"
LORA_RANK="${LORA_RANK:-32}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"

if [[ ! -f "${METADATA_PATH}" ]]; then
  echo "metadata.csv not found. Creating it from images and sidecar .txt captions."
  "${PYTHON_BIN}" /opt/runpod-anima-lora/scripts/make_metadata.py --dataset-dir "${DATASET_DIR}" --output "${METADATA_PATH}"
fi

for required_path in "${CHECKPOINT_PATH}" "${TEXT_ENCODER_PATH}" "${VAE_PATH}" "${METADATA_PATH}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "ERROR: missing required file: ${required_path}" >&2
    exit 3
  fi
done

mkdir -p "${OUTPUT_PATH}"
cd "${DIFFSYNTH_HOME}"

cmd=(
  accelerate launch
  --num_processes "${NUM_PROCESSES}"
  examples/anima/model_training/train.py
  --dataset_base_path "${DATASET_DIR}"
  --dataset_metadata_path "${METADATA_PATH}"
  --data_file_keys "${DATA_FILE_KEYS}"
  --max_pixels "${MAX_PIXELS}"
  --dataset_repeat "${DATASET_REPEAT}"
  --dataset_num_workers "${DATASET_NUM_WORKERS}"
  --model_paths "${MODEL_PATHS_JSON}"
  --tokenizer_path "${TOKENIZER_PATH}"
  --tokenizer_t5xxl_path "${TOKENIZER_T5XXL_PATH}"
  --learning_rate "${LEARNING_RATE}"
  --num_epochs "${NUM_EPOCHS}"
  --remove_prefix_in_ckpt "pipe.dit."
  --output_path "${OUTPUT_PATH}"
  --lora_base_model "dit"
  --lora_target_modules "${LORA_TARGET_MODULES}"
  --lora_rank "${LORA_RANK}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --save_steps "${SAVE_STEPS}"
)

if [[ -n "${HEIGHT:-}" ]]; then
  cmd+=(--height "${HEIGHT}")
fi
if [[ -n "${WIDTH:-}" ]]; then
  cmd+=(--width "${WIDTH}")
fi
if [[ "${USE_GRADIENT_CHECKPOINTING:-1}" == "1" ]]; then
  cmd+=(--use_gradient_checkpointing)
fi
if [[ "${USE_GRADIENT_CHECKPOINTING_OFFLOAD:-0}" == "1" ]]; then
  cmd+=(--use_gradient_checkpointing_offload)
fi
if [[ "${ENABLE_MODEL_CPU_OFFLOAD:-0}" == "1" ]]; then
  cmd+=(--enable_model_cpu_offload)
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
