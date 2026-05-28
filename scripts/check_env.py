#!/usr/bin/env python3
import argparse
import importlib.util
import os
import pathlib
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="/workspace/anima-lora")
    args = parser.parse_args()

    print(f"python: {sys.executable}")
    print(f"python_version: {sys.version.split()[0]}")

    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"cuda_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda_device: {torch.cuda.get_device_name(0)}")
            print(f"cuda_capability: {torch.cuda.get_device_capability(0)}")
    except Exception as exc:
        print(f"torch_check_error: {exc}")

    for module in ["accelerate", "transformers", "safetensors", "huggingface_hub", "bitsandbytes"]:
        print(f"{module}: {'ok' if importlib.util.find_spec(module) else 'missing'}")

    sd_scripts = pathlib.Path(os.environ.get("SD_SCRIPTS_HOME", "/opt/sd-scripts"))
    train_script = sd_scripts / "anima_train_network.py"
    print(f"sd_scripts: {sd_scripts} exists={sd_scripts.exists()}")
    print(f"anima_train_network.py: exists={train_script.exists()}")

    workspace = pathlib.Path(args.workspace)
    for path in [
        workspace / "models" / "diffusion_models",
        workspace / "models" / "text_encoders",
        workspace / "models" / "vae",
        workspace / "datasets" / "train",
        workspace / "outputs",
    ]:
        print(f"path: {path} exists={path.exists()}")


if __name__ == "__main__":
    main()
