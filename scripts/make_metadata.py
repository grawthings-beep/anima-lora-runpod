#!/usr/bin/env python3
import argparse
import csv
import pathlib


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def iter_images(dataset_dir, recursive):
    pattern = "**/*" if recursive else "*"
    for path in sorted(dataset_dir.glob(pattern)):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def read_caption(image_path, default_prompt):
    caption_path = image_path.with_suffix(".txt")
    if caption_path.exists():
        return caption_path.read_text(encoding="utf-8").strip()
    return default_prompt


def main():
    parser = argparse.ArgumentParser(description="Create DiffSynth metadata.csv from images and sidecar .txt captions.")
    parser.add_argument("--dataset-dir", default="/workspace/anima-lora/datasets/train")
    parser.add_argument("--output", default=None)
    parser.add_argument("--default-prompt", default="")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    dataset_dir = pathlib.Path(args.dataset_dir)
    output = pathlib.Path(args.output) if args.output else dataset_dir / "metadata.csv"
    rows = []
    for image_path in iter_images(dataset_dir, args.recursive):
        prompt = read_caption(image_path, args.default_prompt)
        if args.prefix:
            prompt = f"{args.prefix}, {prompt}".strip(", ")
        rows.append(
            {
                "image": image_path.relative_to(dataset_dir).as_posix(),
                "prompt": prompt,
            }
        )

    if not rows:
        raise SystemExit(f"No images found in {dataset_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "prompt"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
