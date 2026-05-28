# RunPod Anima LoRA Trainer (sd-scripts)

RunPod Pod template for training **Anima-native LoRAs** with [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) `anima_train_network.py`.

This is for Anima / WAI-ANIMA. Existing SDXL/Illustrious/ILXL LoRA weights are **not** reusable as Anima LoRA weights; use the same images/captions and retrain.

## Why sd-scripts (not DiffSynth)

- **DiT-only LoRAs load directly in ComfyUI with no conversion.** DiffSynth-Studio output uses different key names, so ComfyUI silently skips the keys (`lora key not loaded`) and the LoRA appears to do nothing. Training DiT-only here (`--network_train_unet_only`) avoids that entirely.
- **No gated dependency.** sd-scripts bundles the T5 tokenizer, so there is no Stability SD3.5 license gate to clear.
- **Faster.** `--cache_latents` + `--cache_text_encoder_outputs` + bf16 skip per-step VAE/text re-encoding.

## Container Image

GitHub Actions builds on push to `main`:

```
ghcr.io/grawthings-beep/anima-lora-runpod:cuda12.8
```

**After the first successful build, set the GHCR package visibility to Public** (Packages -> this package -> Package settings), otherwise RunPod cannot pull it without a registry secret.

## RunPod Template

```
Type: Pod
Compute type: Nvidia GPU (RTX 50-series / Blackwell OK; image is cu12.8 / sm_120-capable)
Container image: ghcr.io/grawthings-beep/anima-lora-runpod:cuda12.8
Volume mount path: /workspace
```

Recommended:

```
Container disk: 40 GB or more
Network Volume: 80 GB or more
```

Environment variables:

```
WORKSPACE_DIR=/workspace/anima-lora
MODEL_ROOT=/workspace/anima-lora
DOWNLOAD_MODELS=1
RUN_DEP_CHECK=1
AUTO_TRAIN=0
HF_TOKEN={{ RUNPOD_SECRET_HF_TOKEN }}
CIVITAI_TOKEN={{ RUNPOD_SECRET_CIVITAI_TOKEN }}
```

Keep tokens in RunPod Secrets. Do not paste raw tokens into a public template.

## Model Layout

Startup downloads models to:

```
/workspace/anima-lora/models/diffusion_models/wai_anima_2859702.safetensors   # DiT (ComfyUI format)
/workspace/anima-lora/models/text_encoders/qwen_3_06b_base.safetensors         # Qwen3-0.6B
/workspace/anima-lora/models/vae/qwen_image_vae.safetensors                    # Qwen-Image VAE
```

The WAI-ANIMA download is Civitai model version `2859702`. To train on the official base instead, enable the `anima-preview3-base` entry in `config/model-manifest.json` and point `DIT_PATH` at it.

## Dataset

Put images and matching `.txt` captions here:

```
/workspace/anima-lora/datasets/train/001.png
/workspace/anima-lora/datasets/train/001.txt
/workspace/anima-lora/datasets/train/002.png
/workspace/anima-lora/datasets/train/002.txt
```

The sd-scripts `dataset.toml` is generated automatically the first time you train. To regenerate it manually:

```
/opt/runpod-anima-lora/scripts/make_metadata.py --dataset-dir /workspace/anima-lora/datasets/train
```

It prints the estimated steps/epoch so you can size the run.

## Train

```
/opt/runpod-anima-lora/scripts/train_lora.sh
```

Output:

```
/workspace/anima-lora/outputs/anima_lora.safetensors
```

Tunable env vars (defaults shown):

```
LORA_RANK=16 LORA_ALPHA=1 LEARNING_RATE=1e-4 \
NUM_REPEATS=10 NUM_EPOCHS=5 RESOLUTION=1024 \
OPTIMIZER_TYPE=AdamW LR_SCHEDULER=constant TIMESTEP_SAMPLING=sigmoid \
/opt/runpod-anima-lora/scripts/train_lora.sh
```

Notes:

- **Aim for ~1,000-2,000 total steps.** Steps/epoch = images x NUM_REPEATS. The old DiffSynth defaults (repeat 50 x epoch 5 = 250 dataset passes) plus no caching were the main cause of ~90-minute runs. With caching + bf16 + a sane step count, expect a much shorter run.
- `LEARNING_RATE=1e-4` is paired with `LORA_ALPHA=1` (the sd-scripts Anima guide value). If you raise alpha, lower the LR.
- `OPTIMIZER_TYPE` defaults to plain `AdamW` for reliability on RTX 50-series. `AdamW8bit` saves VRAM but needs a Blackwell-capable bitsandbytes.
- DiT-only training is hard-set (`--network_train_unet_only`) so the output is ComfyUI-ready and text-encoder outputs can be cached.
- Advanced overrides: `MAX_TRAIN_STEPS`, `BLOCKS_TO_SWAP`, `NETWORK_ARGS` (e.g. `"verbose=True"`), `SAMPLE_PROMPTS`.

## Use In ComfyUI

Copy the trained `.safetensors` LoRA into your ComfyUI Pod:

```
/workspace/comfyui/models/loras/anima
```

Use it with an Anima/WAI-ANIMA workflow. **No conversion needed for DiT-only LoRAs.** If you ever train a Qwen3 text-encoder LoRA, convert it with sd-scripts `networks/convert_anima_lora_to_comfy.py`.

## Sources

- sd-scripts Anima guide: <https://github.com/kohya-ss/sd-scripts/blob/main/docs/anima_train_network.md>
- Anima model and training notes: <https://huggingface.co/circlestone-labs/Anima>
- WAI-ANIMA model page: <https://civitai.com/models/2544636/wai-anima>
