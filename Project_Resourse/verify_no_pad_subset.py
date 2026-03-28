import os
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, roc_auc_score

from config import MALIMG_ARCHIVE_DIR_STR
from models import get_resnet18_grayscale


SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 0
MAX_TRAIN_BATCHES = 200


def _filter_indices_no_pad(image_folder_ds, min_size=256, max_images=None):
    """Keep only images with raw (w,h) >= min_size.

    Returns two lists: (benign_indices, malware_indices) using binary labels.
    """
    warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
    ok_b = []
    ok_m = []
    for i, (path, _y) in enumerate(image_folder_ds.samples):
        if max_images is not None and (len(ok_b) + len(ok_m)) >= max_images:
            break
        try:
            with Image.open(path) as im:
                w, h = im.size
        except Exception:
            continue
        if w >= min_size and h >= min_size:
            raw_t = image_folder_ds.targets[i]
            t = int(image_folder_ds.target_transform(raw_t))
            if t == 0:
                ok_b.append(i)
            else:
                ok_m.append(i)
        if i > 0 and i % 1000 == 0:
            print(f"  scanned={i} kept_b={len(ok_b)} kept_m={len(ok_m)}")
    return ok_b, ok_m


def _count_binary_targets_from_indices(image_folder_ds, indices):
    benign = 0
    malware = 0
    for idx in indices:
        raw_t = image_folder_ds.targets[idx]
        t = int(image_folder_ds.target_transform(raw_t))
        if t == 0:
            benign += 1
        else:
            malware += 1
    return benign, malware


def main(
    max_train_scan=15000,
    max_val_scan=6000,
    min_size=256,
    max_train_keep_per_class=600,
    max_val_keep_per_class=300,
):
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    # No-pad subset: filter to raw (w,h)>=256, then use CenterCrop(256).
    # This avoids introducing padding entirely.
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    base = MALIMG_ARCHIVE_DIR_STR
    train_ds = datasets.ImageFolder(os.path.join(base, "train"), transform=transform)
    val_ds = datasets.ImageFolder(os.path.join(base, "val"), transform=transform)

    # Per-split benign index mapping (must not reuse train index).
    train_benign_idx = train_ds.classes.index("benign")
    val_benign_idx = val_ds.classes.index("benign")
    train_ds.target_transform = lambda x, i=train_benign_idx: 0 if x == i else 1
    val_ds.target_transform = lambda x, i=val_benign_idx: 0 if x == i else 1

    print(f"[*] train benign raw index: {train_benign_idx} ({len(train_ds.classes)} classes)")
    print(f"[*] val   benign raw index: {val_benign_idx} ({len(val_ds.classes)} classes)")

    # Filter out any raw images that are <256 in either dimension
    train_b, train_m = _filter_indices_no_pad(
        train_ds, min_size=min_size, max_images=max_train_scan
    )
    val_b, val_m = _filter_indices_no_pad(
        val_ds, min_size=min_size, max_images=max_val_scan
    )

    n_train_each = min(len(train_b), len(train_m), max_train_keep_per_class)
    n_val_each = min(len(val_b), len(val_m), max_val_keep_per_class)
    if n_train_each == 0 or n_val_each == 0:
        raise RuntimeError("No-pad filter produced an empty class; adjust scan limits/filters.")

    train_idx = random.sample(train_b, n_train_each) + random.sample(train_m, n_train_each)
    val_idx = random.sample(val_b, n_val_each) + random.sample(val_m, n_val_each)
    random.shuffle(train_idx)
    random.shuffle(val_idx)

    train_sub = Subset(train_ds, train_idx)
    val_sub = Subset(val_ds, val_idx)

    b_tr, m_tr = _count_binary_targets_from_indices(train_ds, train_idx)
    b_val, m_val = _count_binary_targets_from_indices(val_ds, val_idx)
    print(f"[*] filtered train: n={len(train_sub)} benign={b_tr} malware={m_tr}")
    print(f"[*] filtered val  : n={len(val_sub)} benign={b_val} malware={m_val}")

    if m_tr == 0 or b_tr == 0:
        raise RuntimeError("Filtered train split has a missing class; adjust filters.")

    pos_weight = torch.tensor([b_tr / m_tr], dtype=torch.float32, device=device)
    print(f"[*] pos_weight (benign/malware): {pos_weight.item():.4f}")

    train_loader = DataLoader(
        train_sub,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_sub,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = get_resnet18_grayscale().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    print("[*] Training 1 epoch on no-pad subset...")
    model.train()
    for bi, (images, labels) in enumerate(train_loader):
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
                pred = (logits > 0).float()
                acc = (pred == labels).float().mean().item()
            print(f"  batch={bi+1} loss={loss.item():.4f} acc={acc:.4f}")

    print("[*] Validating...")
    model.eval()
    correct = 0
    total = 0
    all_probs = []
    all_y = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images).squeeze(1)
            pred = (logits > 0).to(labels.dtype)
            correct += (pred == labels).sum().item()
            total += labels.numel()
            all_probs.append(torch.sigmoid(logits).detach().cpu())
            all_y.append(labels.detach().cpu())

    probs = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_y).numpy().astype(int)
    auc = roc_auc_score(y_true, probs)
    y_pred = (probs >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"[!] no-pad subset val_acc={correct / max(1,total):.4f} val_auc={auc:.4f} (n={total})")
    print("confusion_matrix rows=true [benign,malware], cols=pred [benign,malware]")
    print(cm)


if __name__ == "__main__":
    main()
