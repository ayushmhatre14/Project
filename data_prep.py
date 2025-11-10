# data_prep.py
import os
import random
import shutil
from pathlib import Path

def split_train_val(src_dir, dest_dir, val_split=0.2, seed=123):
    """
    Splits dataset laid out as src_dir/class_name/*.jpg into
    dest_dir/train/class_name and dest_dir/val/class_name.
    """
    random.seed(seed)
    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)
    train_dir = dest_dir / 'train'
    val_dir = dest_dir / 'val'

    for p in [train_dir, val_dir]:
        p.mkdir(parents=True, exist_ok=True)

    classes = [d for d in src_dir.iterdir() if d.is_dir()]
    if not classes:
        raise RuntimeError(f"No class subfolders found in {src_dir}")

    for cls in classes:
        images = list(cls.glob('*'))
        random.shuffle(images)
        n_val = int(len(images) * val_split)
        val_imgs = images[:n_val]
        train_imgs = images[n_val:]

        (train_dir/cls.name).mkdir(exist_ok=True)
        (val_dir/cls.name).mkdir(exist_ok=True)

        for img in train_imgs:
            shutil.copy(img, train_dir/cls.name / img.name)
        for img in val_imgs:
            shutil.copy(img, val_dir/cls.name / img.name)

    print(f"Split done. Train/Val folders created at {dest_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='./data/dataset', help='Source dataset folder with class subfolders')
    parser.add_argument('--out', type=str, default='./data/processed', help='Output folder')
    parser.add_argument('--val', type=float, default=0.2, help='Validation split fraction')
    args = parser.parse_args()
    split_train_val(args.src, args.out, val_split=args.val)
