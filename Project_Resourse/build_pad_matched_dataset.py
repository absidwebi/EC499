import argparse
import os
import random
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from PIL import Image

from config import MALIMG_ARCHIVE_DIR, MALIMG_ARCHIVE_DIR_STR


SEED = 42
NUM_WORKERS = 0

warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)


class PadTo256(object):
    """Match dataset_loader.py PadTo256 behavior (crop then pad right/bottom)."""

    def __call__(self, img):
        import torchvision.transforms.functional as F

        # PIL.Image: prefer width/height; avoids analyzer confusion.
        w = int(getattr(img, "width"))
        h = int(getattr(img, "height"))
        target = 256

        if w > target or h > target:
            img = F.center_crop(img, [target, target])
            w = int(getattr(img, "width"))
            h = int(getattr(img, "height"))

        if w < target or h < target:
            pad_w = target - w
            pad_h = target - h
            img = F.pad(img, [0, 0, pad_w, pad_h], fill=0)

        return img


@dataclass(frozen=True)
class SampleFeat:
    path: str
    rel_class: str
    y: int  # 0 benign, 1 malware
    pad_rows: int
    pad_cols: int
    frac_neg1: float


def _pad_area(pad_rows: int, pad_cols: int) -> int:
    # PadTo256 pads right/bottom, so padded area is bottom full rows plus
    # right full cols excluding the bottom-right overlap.
    return int(pad_rows) * 256 + int(pad_cols) * (256 - int(pad_rows))


class ImageFolderWithPath(Dataset):
    def __init__(self, root: str, transform):
        self.ds = datasets.ImageFolder(root, transform=transform)
        if "benign" not in self.ds.classes:
            raise ValueError(f"Missing 'benign' in {root}; classes={self.ds.classes}")
        self.benign_idx = self.ds.classes.index("benign")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        path, raw_t = self.ds.samples[idx]
        img, _ = self.ds[idx]
        y = 0 if raw_t == self.benign_idx else 1
        rel_class = os.path.basename(os.path.dirname(path))
        return img, y, path, rel_class


def _extract_feats(split_dir: str, batch_size: int, max_images: int | None) -> List[SampleFeat]:
    tfm = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            PadTo256(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    ds = ImageFolderWithPath(split_dir, transform=tfm)

    # Important: ImageFolder orders samples by class then filename.
    # If we cap by "first N", we bias toward early classes. Use a random
    # subset when max_images is set.
    if max_images is not None and max_images > 0 and max_images < len(ds):
        idx = list(range(len(ds)))
        random.shuffle(idx)
        ds = Subset(ds, idx[:max_images])

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    feats: List[SampleFeat] = []
    seen = 0
    for bi, (imgs, ys, paths, rel_classes) in enumerate(loader):
        x = imgs.squeeze(1)
        is_neg1 = (x == -1.0)
        pad_rows = is_neg1.all(dim=2).sum(dim=1).cpu().numpy().astype(np.int64)
        pad_cols = is_neg1.all(dim=1).sum(dim=1).cpu().numpy().astype(np.int64)
        ys_np = np.asarray(ys, dtype=np.int64)

        frac_neg1 = is_neg1.float().mean(dim=(1, 2)).cpu().numpy().astype(np.float64)

        for i in range(imgs.size(0)):
            feats.append(
                SampleFeat(
                    path=str(paths[i]),
                    rel_class=str(rel_classes[i]),
                    y=int(ys_np[i]),
                    pad_rows=int(pad_rows[i]),
                    pad_cols=int(pad_cols[i]),
                    frac_neg1=float(frac_neg1[i]),
                )
            )

        seen += imgs.size(0)
        if bi % 10 == 0:
            print(f"  batches={bi} images={seen}")
        # when using Subset(max_images), loader ends naturally
    return feats


def _select_pad_matched(
    feats: List[SampleFeat],
    bin_w_rows: int,
    bin_w_cols: int,
    bin_w_frac: float,
    max_per_bin: int,
) -> List[SampleFeat]:
    # Bin key includes frac_neg1 to avoid matching only row/col while leaving
    # overall -1 pixel fraction as a shortcut.
    bins: Dict[Tuple[int, int, int], Dict[int, List[SampleFeat]]] = {}
    for s in feats:
        rb = int(s.pad_rows) // bin_w_rows
        cb = int(s.pad_cols) // bin_w_cols
        fb = int(float(s.frac_neg1) / max(1e-9, float(bin_w_frac)))
        k = (rb, cb, fb)
        if k not in bins:
            bins[k] = {0: [], 1: []}
        bins[k][s.y].append(s)

    selected: List[SampleFeat] = []
    possible = 0
    overlap_bins = 0
    for k in sorted(bins.keys()):
        b_list = bins[k][0]
        m_list = bins[k][1]
        n = min(len(b_list), len(m_list))
        if n <= 0:
            continue
        overlap_bins += 1
        possible += n
        n = min(n, max_per_bin)
        selected.extend(random.sample(b_list, n))
        selected.extend(random.sample(m_list, n))

    random.shuffle(selected)
    print(f"[*] overlap_bins={overlap_bins} possible_matched_pairs={possible}")
    return selected


def _copy_selected(selected: List[SampleFeat], dst_split_dir: str):
    os.makedirs(dst_split_dir, exist_ok=True)
    copied = 0
    for s in selected:
        dst_class_dir = os.path.join(dst_split_dir, s.rel_class)
        os.makedirs(dst_class_dir, exist_ok=True)
        dst_path = os.path.join(dst_class_dir, os.path.basename(s.path))
        shutil.copy2(s.path, dst_path)
        copied += 1
    print(f"[*] Copied {copied} files -> {dst_split_dir}")


def main(
    out_root: str = str(MALIMG_ARCHIVE_DIR.parent / "malimg_dataset_padmatched"),
    batch_size: int = 512,
    max_train_images: int | None = None,
    max_val_images: int | None = None,
    max_test_images: int | None = None,
    bin_w_rows: int = 4,
    bin_w_cols: int = 4,
    bin_w_frac: float = 0.02,
    max_per_bin_train: int = 200,
    max_per_bin_eval: int = 100,
):
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    src_root = Path(MALIMG_ARCHIVE_DIR_STR)
    dst_root = Path(out_root)
    print(f"[*] Source dataset: {src_root}")
    print(f"[*] Output dataset: {dst_root}")

    if dst_root.exists():
        print(f"[*] Removing existing: {dst_root}")
        shutil.rmtree(dst_root)

    for split, max_images, max_per_bin in [
        ("train", max_train_images, max_per_bin_train),
        ("val", max_val_images, max_per_bin_eval),
        ("test", max_test_images, max_per_bin_eval),
    ]:
        print(f"\n== Building pad-matched {split} ==")
        split_src = str(src_root / split)
        feats = _extract_feats(split_src, batch_size=batch_size, max_images=max_images)
        print(f"[*] extracted n={len(feats)}")

        sel = _select_pad_matched(
            feats,
            bin_w_rows=bin_w_rows,
            bin_w_cols=bin_w_cols,
            bin_w_frac=bin_w_frac,
            max_per_bin=max_per_bin,
        )
        n_b = sum(1 for s in sel if s.y == 0)
        n_m = sum(1 for s in sel if s.y == 1)
        print(f"[*] selected n={len(sel)} benign={n_b} malware={n_m}")

        if len(sel) > 0:
            # quick sanity: show mean padding area by class
            b_area = [_pad_area(s.pad_rows, s.pad_cols) for s in sel if s.y == 0]
            m_area = [_pad_area(s.pad_rows, s.pad_cols) for s in sel if s.y == 1]
            print(f"[*] pad_area_mean benign={np.mean(b_area):.1f} malware={np.mean(m_area):.1f}")
            b_frac = [s.frac_neg1 for s in sel if s.y == 0]
            m_frac = [s.frac_neg1 for s in sel if s.y == 1]
            print(f"[*] frac_-1_mean benign={np.mean(b_frac):.4f} malware={np.mean(m_frac):.4f}")

        split_dst = str(dst_root / split)
        _copy_selected(sel, split_dst)

    print("\n[!] Done. Pad-matched dataset written.")
    print(f"    {dst_root}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build pad-matched subset dataset")
    p.add_argument(
        "--out-root",
        default=str(MALIMG_ARCHIVE_DIR.parent / "malimg_dataset_padmatched"),
        help="Output dataset root containing train/val/test",
    )
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument(
        "--max-train-images",
        type=int,
        default=0,
        help="0 means use full split; otherwise random subset size",
    )
    p.add_argument(
        "--max-val-images",
        type=int,
        default=0,
        help="0 means use full split; otherwise random subset size",
    )
    p.add_argument(
        "--max-test-images",
        type=int,
        default=0,
        help="0 means use full split; otherwise random subset size",
    )
    p.add_argument("--bin-w-rows", type=int, default=4)
    p.add_argument("--bin-w-cols", type=int, default=4)
    p.add_argument("--bin-w-frac", type=float, default=0.02)
    p.add_argument("--max-per-bin-train", type=int, default=200)
    p.add_argument("--max-per-bin-eval", type=int, default=100)
    args = p.parse_args()

    def nz(x):
        return None if x == 0 else x

    main(
        out_root=args.out_root,
        batch_size=args.batch_size,
        max_train_images=nz(args.max_train_images),
        max_val_images=nz(args.max_val_images),
        max_test_images=nz(args.max_test_images),
        bin_w_rows=args.bin_w_rows,
        bin_w_cols=args.bin_w_cols,
        bin_w_frac=args.bin_w_frac,
        max_per_bin_train=args.max_per_bin_train,
        max_per_bin_eval=args.max_per_bin_eval,
    )
