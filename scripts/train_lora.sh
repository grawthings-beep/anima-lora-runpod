#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="$(command -v python || command -v python3 || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "ERROR: neither python nor python3 was found in PATH." >&2
  exit 2
fi

SD_SCRIPTS_HOME="${SD_SCRIPTS_HOME:-/opt/sd-scripts}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace/anima-lora}"
MODEL_ROOT="${MODEL_ROOT:-${WORKSPACE_DIR}}"
DATASET_DIR="${DATASET_DIR:-${WORKSPACE_DIR}/datasets/train}"
DATASET_CONFIG="${DATASET_CONFIG:-${WORKSPACE_DIR}/datasets/dataset.toml}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_DIR}/outputs}"
OUTPUT_NAME="${OUTPUT_NAME:-anima_lora}"

# Model files (same three Anima components)
DIT_PATH="${DIT_PATH:-${MODEL_ROOT}/models/diffusion_models/wai_anima_2859702.safetensors}"
QWEN3_PATH="${QWEN3_PATH:-${MODEL_ROOT}/models/text_encoders/qwen_3_06b_base.safetensors}"
VAE_PATH="${VAE_PATH:-${MODEL_ROOT}/models/vae/qwen_image_vae.safetensors}"

# Training hyperparameters (defaults follow the sd-scripts Anima guide)
RESOLUTION="${RESOLUTION:-1024}"
NUM_REPEATS="${NUM_REPEATS:-10}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"   # paired with LORA_ALPHA=1 (see README)
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-1}"
OPTIMIZER_TYPE="${OPTIMIZER_TYPE:-AdamW}"  # AdamW8bit needs a Blackwell-capable bitsandbytes
LR_SCHEDULER="${LR_SCHEDULER:-constant}"
TIMESTEP_SAMPLING="${TIMESTEP_SAMPLING:-sigmoid}"
SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
SEED="${SEED:-42}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MAX_DATA_LOADER_N_WORKERS="${MAX_DATA_LOADER_N_WORKERS:-2}"

if [[ ! -f "${DATASET_CONFIG}" ]]; then
  echo "dataset.toml not found. Generating from images in ${DATASET_DIR}."
  "${PYTHON_BIN}" /opt/runpod-anima-lora/scripts/make_metadata.py \
    --dataset-dir "${DATASET_DIR}" \
    --output "${DATASET_CONFIG}" \
    --resolution "${RESOLUTION}" \
    --num-repeats "${NUM_REPEATS}"
fi

for required_path in "${DIT_PATH}" "${QWEN3_PATH}" "${VAE_PATH}" "${DATASET_CONFIG}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "ERROR: missing required file: ${required_path}" >&2
    exit 3
  fi
done

mkdir -p "${OUTPUT_DIR}"
cd "${SD_SCRIPTS_HOME}"

cmd=(
  accelerate launch
  --num_cpu_threads_per_process 1
  --num_processes "${NUM_PROCESSES}"
  anima_train_network.py
  --pretrained_model_name_or_path "${DIT_PATH}"
  --qwen3 "${QWEN3_PATH}"
  --vae "${VAE_PATH}"
  --dataset_config "${DATASET_CONFIG}"
  --output_dir "${OUTPUT_DIR}"
  --output_name "${OUTPUT_NAME}"
  --save_model_as safetensors
  --network_module networks.lora_anima
  --network_dim "${LORA_RANK}"
  --network_alpha "${LORA_ALPHA}"
  --learning_rate "${LEARNING_RATE}"
  --optimizer_type "${OPTIMIZER_TYPE}"
  --lr_scheduler "${LR_SCHEDULER}"
  --timestep_sampling "${TIMESTEP_SAMPLING}"
  --max_train_epochs "${NUM_EPOCHS}"
  --save_every_n_epochs "${SAVE_EVERY_N_EPOCHS}"
  --mixed_precision "${MIXED_PRECISION}"
  --gradient_checkpointing
  --cache_latents
  --cache_latents_to_disk
  --cache_text_encoder_outputs
  --cache_text_encoder_outputs_to_disk
  --network_train_unet_only
  --max_data_loader_n_workers "${MAX_DATA_LOADER_N_WORKERS}"
  --seed "${SEED}"
)

# Optional overrides
if [[ -n "${MAX_TRAIN_STEPS:-}" ]]; then
  cmd+=(--max_train_steps "${MAX_TRAIN_STEPS}")
fi
if [[ -n "${BLOCKS_TO_SWAP:-}" ]]; then
  cmd+=(--blocks_to_swap "${BLOCKS_TO_SWAP}")
fi
if [[ -n "${NETWORK_ARGS:-}" ]]; then
  # space-separated key=value pairs, e.g. NETWORK_ARGS="verbose=True rank_dropout=0.1"
  # shellcheck disable=SC2206
  extra_args=(${NETWORK_ARGS})
  cmd+=(--network_args "${extra_args[@]}")
fi
if [[ -n "${SAMPLE_PROMPTS:-}" ]]; then
  cmd+=(--sample_prompts "${SAMPLE_PROMPTS}" --sample_every_n_epochs "${SAMPLE_EVERY_N_EPOCHS:-1}")
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
