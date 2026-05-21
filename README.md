# RunPod Anima LoRA Trainer

RunPod Pod template for training Anima-native LoRAs with DiffSynth-Studio.

This is for Anima/WAI-ANIMA. Existing SDXL/Illustrious/ILXL LoRA weights are not reusable as Anima LoRA weights; use the same images/captions and retrain.

## Container Image

After pushing this repo to GitHub, GitHub Actions builds:

```text
ghcr.io/YOUR_GITHUB_USER/YOUR_REPO:cuda12.8
```

For your account, it will look like:

```text
ghcr.io/grawthings-beep/anima-lora-runpod:cuda12.8
```

## RunPod Template

Use:

```text
Type: Pod
Compute type: Nvidia GPU
Container image: ghcr.io/YOUR_GITHUB_USER/YOUR_REPO:cuda12.8
Volume mount path: /workspace
```

Recommended:

```text
Container disk: 40 GB or more
Network Volume: 80 GB or more
```

Environment variables:

```text
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

```text
/workspace/anima-lora/models/diffusion_models/wai_anima_2859702.safetensors
/workspace/anima-lora/models/text_encoders/qwen_3_06b_base.safetensors
/workspace/anima-lora/models/vae/qwen_image_vae.safetensors
```

The WAI-ANIMA download is the Civitai model version `2859702`.

## Dataset

Put images and matching `.txt` captions here:

```text
/workspace/anima-lora/datasets/train
```

Example:

```text
001.png
001.txt
002.png
002.txt
```

Create metadata:

```bash
/opt/runpod-anima-lora/scripts/make_metadata.py --dataset-dir /workspace/anima-lora/datasets/train
```

## Train

Run:

```bash
/opt/runpod-anima-lora/scripts/train_lora.sh
```

Output goes to:

```text
/workspace/anima-lora/outputs/anima_lora
```

Useful overrides:

```bash
LEARNING_RATE=1e-4 NUM_EPOCHS=5 DATASET_REPEAT=50 LORA_RANK=32 /opt/runpod-anima-lora/scripts/train_lora.sh
```

## Use In ComfyUI

Copy the trained `.safetensors` LoRA into your ComfyUI Pod:

```text
/workspace/comfyui/models/loras/anima
```

Then use it with an Anima-compatible ComfyUI workflow and WAI-ANIMA/Anima model, not with ILXL checkpoints.

## Sources

- Anima model and training notes: https://huggingface.co/circlestone-labs/Anima
- WAI-ANIMA model page: https://civitai.red/models/2544636/wai-anima?modelVersionId=2859702
- DiffSynth-Studio Anima docs: https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/en/Model_Details/Anima.md
