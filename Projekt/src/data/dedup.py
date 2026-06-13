import shutil
import hashlib
from collections import defaultdict
from PIL import Image
from pathlib import Path
from src.config import DATA_DIR


# Dataset had both capsicum and bell_pepper classes which is the same plant, and images were duplicated too.
def merge_classes(src_class: str, dst_class: str):
    for split in ["train", "val"]:
        src = DATA_DIR / split / src_class
        dst = DATA_DIR / split / dst_class
        for img in src.iterdir():
            shutil.move(img, dst / img.name)
        src.rmdir()


def file_hash(path):
    with Image.open(path) as img:
        return hashlib.md5(img.convert("RGB").tobytes()).hexdigest()


# merge_classes("corn", "sweetcorn")
# merge_classes("capsicum", "bell pepper")

hashes = defaultdict(list)
for p in DATA_DIR.rglob("*.jpg"):
    if p.is_file():
        hashes[file_hash(p)].append(p)

duplicates = {h: paths for h, paths in hashes.items() if len(paths) > 1}

for h, paths in duplicates.items():
    val_copies = [p for p in paths if "/val/" in str(p)]
    train_copies = [p for p in paths if "/train/" in str(p)]

    if val_copies:
        for p in val_copies[1:]:
            p.unlink()
        for p in train_copies:
            p.unlink()
    else:
        for p in train_copies[1:]:
            p.unlink()


for split in ["train", "val"]:
    print(split)
    for d in sorted((DATA_DIR / split).iterdir()):
        if d.is_dir():
            print(f"  {d.name}: {len(list(d.glob('*.jpg')))}")
