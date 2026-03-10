#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Windows-safe SDXL LoRA fine-tuning launcher using Hugging Face Diffusers.

This version avoids the most common Windows issues:
- no xformers by default
- no 8-bit Adam / bitsandbytes
- conservative training launch
- basic environment checks

Expected dataset structure:
my_dataset/
├── image_001.jpg
├── image_001.txt
├── image_002.png
├── image_002.txt
└── ...

Each .txt file contains the caption for the image with the same base name.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def run(cmd: list[str], cwd: str | None = None) -> None:
    print("\n[RUN]", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def check_python() -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10+ is recommended.")


def ensure_tool_exists(tool_name: str) -> None:
    if shutil.which(tool_name) is None:
        raise RuntimeError(f"Required tool not found in PATH: {tool_name}")


def clone_or_update_diffusers(target_dir: Path) -> Path:
    repo_dir = target_dir / "diffusers"

    if repo_dir.exists():
        print(f"[INFO] Updating existing diffusers repo: {repo_dir}")
        run(["git", "pull"], cwd=str(repo_dir))
    else:
        print(f"[INFO] Cloning diffusers into: {repo_dir}")
        run(["git", "clone", "https://github.com/huggingface/diffusers.git", str(repo_dir)])

    return repo_dir


def install_requirements(repo_dir: Path) -> None:
    print("[INFO] Installing/updating required packages")
    run([sys.executable, "-m", "pip", "install", "-U", "pip"])
    run([sys.executable, "-m", "pip", "install", "-e", "."], cwd=str(repo_dir))

    req_file = repo_dir / "examples" / "text_to_image" / "requirements_sdxl.txt"
    run([sys.executable, "-m", "pip", "install", "-r", str(req_file)])


def write_accelerate_config() -> None:
    print("[INFO] Ensuring Accelerate config exists")
    code = (
        "from pathlib import Path\n"
        "from accelerate.utils import write_basic_config\n"
        "cfg = Path.home() / '.cache' / 'huggingface' / 'accelerate' / 'default_config.yaml'\n"
        "if cfg.exists():\n"
        "    print('Accelerate config already exists')\n"
        "else:\n"
        "    write_basic_config(mixed_precision='fp16')\n"
        "    print('Accelerate config created')\n"
    )
    run([sys.executable, "-c", code])


def validate_dataset(dataset_dir: Path) -> list[Path]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    images = [p for p in dataset_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

    if not images:
        raise RuntimeError(f"No supported image files found in: {dataset_dir}")

    missing_txt = []
    empty_txt = []

    for img in images:
        txt = img.with_suffix(".txt")
        if not txt.exists():
            missing_txt.append(img.name)
        else:
            content = txt.read_text(encoding="utf-8").strip()
            if not content:
                empty_txt.append(txt.name)

    if missing_txt:
        preview = ", ".join(missing_txt[:10])
        raise RuntimeError(
            "Missing caption files for images: "
            f"{preview}" + (" ..." if len(missing_txt) > 10 else "")
        )

    if empty_txt:
        preview = ", ".join(empty_txt[:10])
        raise RuntimeError(
            "Empty caption files found: "
            f"{preview}" + (" ..." if len(empty_txt) > 10 else "")
        )

    print(f"[INFO] Found {len(images)} images with valid matching captions.")
    return sorted(images)


def prepare_hf_dataset(dataset_dir: Path, prepared_root: Path) -> Path:
    """
    Build a Hugging Face imagefolder-style training dataset:

    prepared_root/
      train/
        metadata.jsonl
        image_001.jpg
        image_002.png
        ...
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prepared_root = prepared_root.parent / f"{prepared_root.name}_{timestamp}"

    train_dir = prepared_root / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    images = validate_dataset(dataset_dir)
    metadata_path = train_dir / "metadata.jsonl"

    with metadata_path.open("w", encoding="utf-8") as f:
        for img in images:
            caption = img.with_suffix(".txt").read_text(encoding="utf-8").strip()

            dst = train_dir / img.name
            shutil.copy2(img, dst)

            record = {
                "file_name": img.name,
                "text": caption
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[INFO] Prepared dataset at: {train_dir}")
    return prepared_root


def print_env_debug() -> None:
    print("[INFO] Python executable:", sys.executable)

    try:
        code = (
            "import torch\n"
            "print('torch:', torch.__version__)\n"
            "print('cuda available:', torch.cuda.is_available())\n"
            "print('cuda version:', torch.version.cuda)\n"
            "print('device count:', torch.cuda.device_count())\n"
            "print('device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')\n"
        )
        run([sys.executable, "-c", code])
    except Exception as e:
        print(f"[WARN] Could not print torch environment info: {e}")


def launch_training(
    repo_dir: Path,
    prepared_dataset_dir: Path,
    output_dir: Path,
    model_name: str,
    vae_name: str,
    resolution: int,
    batch_size: int,
    grad_accum: int,
    learning_rate: float,
    max_steps: int,
    rank: int,
    train_text_encoder: bool,
    validation_prompt: str | None,
    validation_epochs: int,
    checkpointing_steps: int,
) -> None:
    train_script = repo_dir / "examples" / "text_to_image" / "train_text_to_image_lora_sdxl.py"

    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")

    cmd = [
        "accelerate",
        "launch",
        str(train_script),
        "--pretrained_model_name_or_path", model_name,
        "--pretrained_vae_model_name_or_path", vae_name,
        "--train_data_dir", str(prepared_dataset_dir),
        "--image_column", "image",
        "--caption_column", "text",
        "--resolution", str(resolution),
        "--train_batch_size", str(batch_size),
        "--gradient_accumulation_steps", str(grad_accum),
        "--gradient_checkpointing",
        "--max_train_steps", str(max_steps),
        "--learning_rate", str(learning_rate),
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0",
        "--snr_gamma", "5.0",
        "--checkpointing_steps", str(checkpointing_steps),
        "--mixed_precision", "fp16",
        "--output_dir", str(output_dir),
        "--rank", str(rank),
    ]

    # Optional but usually okay. Remove if you want perfectly deterministic behavior.
    cmd.append("--random_flip")

    if train_text_encoder:
        cmd.append("--train_text_encoder")

    if validation_prompt:
        cmd.extend([
            "--validation_prompt", validation_prompt,
            "--num_validation_images", "2",
            "--validation_epochs", str(validation_epochs),
        ])

    run(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Windows-safe SDXL LoRA fine-tuning")

    parser.add_argument("--dataset_dir", type=Path, required=True, help="Folder with images + matching .txt captions")
    parser.add_argument("--workspace", type=Path, default=Path("./sdxl_lora_workspace"))
    parser.add_argument("--output_dir", type=Path, default=Path("./my_sdxl_lora"))

    parser.add_argument("--model_name", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--vae_name", type=str, default="madebyollin/sdxl-vae-fp16-fix")

    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--checkpointing_steps", type=int, default=500)

    parser.add_argument("--train_text_encoder", action="store_true")
    parser.add_argument("--validation_prompt", type=str, default="epfl_ecal_lab_style portrait of a person, studio photography")
    parser.add_argument("--validation_epochs", type=int, default=1)

    return parser.parse_args()


def main() -> None:
    print("launch")
    args = parse_args()

    check_python()
    ensure_tool_exists("git")
    ensure_tool_exists("accelerate")

    args.workspace.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print_env_debug()

    print("[INFO] Validating dataset")
    validate_dataset(args.dataset_dir)

    print("[INFO] Cloning/updating diffusers")
    repo_dir = clone_or_update_diffusers(args.workspace)

    print("[INFO] Installing requirements")
    install_requirements(repo_dir)

    print("[INFO] Writing accelerate config")
    write_accelerate_config()

    print("[INFO] Preparing dataset")
    prepared_dataset_dir = prepare_hf_dataset(
        dataset_dir=args.dataset_dir,
        prepared_root=args.workspace / "prepared_dataset"
    )

    print("[INFO] Launching SDXL LoRA training")
    launch_training(
        repo_dir=repo_dir,
        prepared_dataset_dir=prepared_dataset_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        vae_name=args.vae_name,
        resolution=args.resolution,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        rank=args.rank,
        train_text_encoder=args.train_text_encoder,
        validation_prompt=args.validation_prompt.strip() if args.validation_prompt else None,
        validation_epochs=args.validation_epochs,
        checkpointing_steps=args.checkpointing_steps,
    )

    print("\n[DONE] Training completed.")
    print(f"[DONE] Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()