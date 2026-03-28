import os
import random
import warnings
from typing import Any, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image

from config import MALIMG_ARCHIVE_DIR_STR
from models import get_resnet18_grayscale


SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 0
MAX_TRAIN_BATCHES = 250


class PadTo256(object):
    def __call__(self, img):
        # Not used in this script; kept only to mirror main pipeline transform.
        return img


def make_dataset(split):
    tfm = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            # Use the exact PadTo256 defined in dataset_loader (crop if >256, else pad right/bottom).
            transforms.Lambda(lambda img: _pad_to_256(img)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    ds = datasets.ImageFolder(os.path.join(MALIMG_ARCHIVE_DIR_STR, split), transform=tfm)
    if "benign" not in ds.classes:
        raise ValueError(f"split={split} missing 'benign' folder; classes={ds.classes}")
    benign_idx = ds.classes.index("benign")
    ds.target_transform = lambda x, i=benign_idx: 0 if x == i else 1
    print(f"[*] {split}: benign raw index {benign_idx} ({len(ds.classes)} classes)")
    return ds


def _pad_to_256(img):
    import torchvision.transforms.functional as F


    def _get_wh(x: Any) -> Tuple[int, int]:
        if hasattr(x, "width") and hasattr(x, "height"):
            return int(getattr(x, "width")), int(getattr(x, "height"))
        if hasattr(x, "shape"):
            shp = cast(Any, getattr(x, "shape"))
            return int(shp[-1]), int(shp[-2])
        sz = getattr(x, "size")
        sz = sz() if callable(sz) else sz
        sz_t = cast(Tuple[int, int], sz)
        return int(sz_t[0]), int(sz_t[1])

    w, h = _get_wh(img)
    target = 256

    if w > target or h > target:
        img = F.center_crop(img, [target, target])

    w, h = _get_wh(img)
    if w < target or h < target:
        pad_w = target - w
        pad_h = target - h
        img = F.pad(img, [0, 0, pad_w, pad_h], fill=0)

    return img


@torch.no_grad()
def compute_pad_cols_from_raw_sizes(ds, max_scan=None):
    """Compute pad_cols implied by PadTo256 using raw PNG (w,h).

    PadTo256 behavior:
      - if w>256 or h>256: center_crop to 256 -> no padding
      - else: pad right/bottom to 256

    Therefore pad_cols = max(0, 256 - w) if w<256 else 0.
    """
    warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
    idxs = []
    ys = []
    pad_cols = []

    for i, (path, _raw_t) in enumerate(ds.samples):
        if max_scan is not None and i >= max_scan:
            break
        try:
            with Image.open(path) as im:
                w, h = im.size
        except Exception:
            continue

        raw_t = ds.targets[i]
        y = int(ds.target_transform(raw_t))
        pc = 0
        if w < 256 and h <= 256:
            pc = 256 - int(w)
        elif w < 256 and h < 256:
            pc = 256 - int(w)
        else:
            pc = 0

        idxs.append(i)
        ys.append(y)
        pad_cols.append(int(pc))

        if i > 0 and i % 5000 == 0:
            print(f"  scanned={i}")

    return (
        np.asarray(idxs, dtype=np.int64),
        np.asarray(ys, dtype=np.int64),
        np.asarray(pad_cols, dtype=np.int64),
    )


def build_pad_matched_indices(idx, y, pad_cols, bin_width=8, max_per_bin=200):
    # bin by pad_cols, sample equal counts in each bin
    bins = (pad_cols // bin_width).astype(np.int64)
    matched = []
    for b in np.unique(bins):
        in_bin = (bins == b)
        b_idx = idx[in_bin & (y == 0)]
        m_idx = idx[in_bin & (y == 1)]
        n = min(len(b_idx), len(m_idx))
        if n <= 0:
            continue
        n = min(n, max_per_bin)
        matched.extend(random.sample(b_idx.tolist(), n))
        matched.extend(random.sample(m_idx.tolist(), n))
    random.shuffle(matched)
    return matched


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for bi, (images, labels) in enumerate(loader):
        if bi >= MAX_TRAIN_BATCHES:
            print(f"[*] stopping early at batch={bi} (MAX_TRAIN_BATCHES={MAX_TRAIN_BATCHES})")
            break
        images = images.to(device)
        labels = labels.float().to(device).unsqueeze(1)
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (bi + 1) % 50 == 0:
            with torch.no_grad():
                acc = ((logits > 0).float() == labels).float().mean().item()
            print(f"  batch={bi+1} loss={loss.item():.4f} acc={acc:.4f}")


@torch.no_grad()
def eval_split(model, loader, device):
    model.eval()
    probs = []
    y_true = []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images).squeeze(1)
        probs.append(torch.sigmoid(logits).cpu())
        y_true.append(labels.cpu())
    p = torch.cat(probs).numpy()
    y = torch.cat(y_true).numpy().astype(int)
    y_pred = (p >= 0.5).astype(int)
    acc = float((y_pred == y).mean())
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    return acc, auc, cm


def main(train_scan=12000, val_scan=5000, bin_width=8, max_per_bin=200):
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    train_ds = make_dataset("train")
    val_ds = make_dataset("val")

    print("[*] computing pad_cols per sample...")
    tr_idx, tr_y, tr_pad = compute_pad_cols_from_raw_sizes(train_ds, max_scan=train_scan)
    va_idx, va_y, va_pad = compute_pad_cols_from_raw_sizes(val_ds, max_scan=val_scan)

    tr_sel = build_pad_matched_indices(
        tr_idx, tr_y, tr_pad, bin_width=bin_width, max_per_bin=max_per_bin
    )
    va_sel = build_pad_matched_indices(
        va_idx, va_y, va_pad, bin_width=bin_width, max_per_bin=max_per_bin // 2
    )
    print(f"[*] pad-matched train n={len(tr_sel)}")
    print(f"[*] pad-matched val   n={len(va_sel)}")

    if len(tr_sel) < 200 or len(va_sel) < 200:
        raise RuntimeError("Pad-matched subset too small; adjust binning parameters.")

    train_sub = Subset(train_ds, tr_sel)
    val_sub = Subset(val_ds, va_sel)

    # balanced subset -> pos_weight ~ 1
    pos_weight = torch.tensor([1.0], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = get_resnet18_grayscale().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    print("[*] training 1 epoch on pad-matched subset...")
    train_one_epoch(model, train_loader, criterion, optimizer, device)

    print("[*] evaluating...")
    acc, auc, cm = eval_split(model, val_loader, device)
    print(f"[!] pad-matched subset val_acc={acc:.4f} val_auc={auc:.4f} (n={len(val_sub)})")
    print("confusion_matrix rows=true [benign,malware], cols=pred [benign,malware]")
    print(cm)


if __name__ == "__main__":
    main()
