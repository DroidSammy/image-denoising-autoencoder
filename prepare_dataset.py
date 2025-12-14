# prepare_dataset.py
import os, shutil, random
from pathlib import Path
from PIL import Image

SRC_ROOT = Path("data/SIDD/SIDD_medium")
SRC_NOISY = SRC_ROOT / "Source-patches"
SRC_GT = SRC_ROOT / "Target-patches"

OUT_ROOT = Path("data")
TRAIN_DIR = OUT_ROOT / "train"
VAL_DIR = OUT_ROOT / "val"

IMAGE_SIZE = (256, 256)
TRAIN_SPLIT = 0.9
VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def list_images(folder):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in VALID_EXTS and p.is_file()])

def prepare_out_dirs():
    for d in (TRAIN_DIR, VAL_DIR):
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

def copy_and_resize(src: Path, dst: Path):
    im = Image.open(src).convert("RGB")
    im = im.resize(IMAGE_SIZE, Image.BICUBIC)
    im.save(dst, format="PNG")

def main():
    if not SRC_NOISY.exists() or not SRC_GT.exists():
        print("Source or Target patches not found. Check:", SRC_NOISY, SRC_GT); return

    noisy_files = list_images(SRC_NOISY)
    gt_files = list_images(SRC_GT)
    noisy_map = {p.stem: p for p in noisy_files}
    gt_map = {p.stem: p for p in gt_files}

    pairs = []
    for key, noisy_path in noisy_map.items():
        if key in gt_map:
            pairs.append((noisy_path, gt_map[key]))
        else:
            short_key = key.split("_")[0]
            for gk, gp in gt_map.items():
                if gk.startswith(short_key):
                    pairs.append((noisy_path, gp)); break

    print(f"Found {len(pairs)} matching pairs.")
    if len(pairs) == 0:
        print("No pairs found. Check filenames."); return

    random.shuffle(pairs)
    split_idx = int(len(pairs) * TRAIN_SPLIT)
    train_pairs, val_pairs = pairs[:split_idx], pairs[split_idx:]

    prepare_out_dirs()
    for i, (n, g) in enumerate(train_pairs):
        copy_and_resize(n, TRAIN_DIR / f"noisy_{i:06d}.png")
        copy_and_resize(g, TRAIN_DIR / f"gt_{i:06d}.png")
    for i, (n, g) in enumerate(val_pairs):
        copy_and_resize(n, VAL_DIR / f"noisy_{i:06d}.png")
        copy_and_resize(g, VAL_DIR / f"gt_{i:06d}.png")
    print("Done preparing dataset.")

if __name__ == "__main__":
    main()
