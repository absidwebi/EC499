import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.linear_model import LogisticRegression

from dataset_loader import get_data_loaders
from models import get_resnet18_grayscale
from config import MALIMG_ARCHIVE_DIR_STR, LOGS_DIR


# Train on full dataset but neutralize padding-feature shortcut via sampling.

SEED = 42
DATA_DIR = MALIMG_ARCHIVE_DIR_STR

BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
EARLY_STOP_PATIENCE = 5
NUM_WORKERS = 0

WEIGHT_CLIP_MAX = 20.0


def _suppress_pil_decompression_bomb_warnings():
    try:
        from PIL import Image
        import warnings

        warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
    except Exception:
        pass


@torch.no_grad()
def _extract_padding_features(loader):
    # Features based on normalized tensors from the existing pipeline.
    X = []
    y = []
    for imgs, lbls in loader:
        x = imgs.squeeze(1)
        is_neg1 = (x == -1.0)
        pad_rows = is_neg1.all(dim=2).sum(dim=1).cpu().numpy()
        pad_cols = is_neg1.all(dim=1).sum(dim=1).cpu().numpy()
        frac = is_neg1.float().mean(dim=(1, 2)).cpu().numpy()
        X.append(np.stack([pad_rows, pad_cols, frac], axis=1))
        y.append(lbls.cpu().numpy().astype(np.int64))
    return np.concatenate(X, axis=0), np.concatenate(y, axis=0)


def _compute_propensity_weights(X, y):
    # Fit a simple model that predicts label from padding features.
    # Then weight samples by inverse propensity for their true class.
    clf = LogisticRegression(max_iter=4000, class_weight="balanced")
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    p_true = proba[np.arange(len(y)), y]
    w = 1.0 / np.maximum(p_true, 1e-6)
    w = np.minimum(w, WEIGHT_CLIP_MAX)
    # Normalize for readability only; sampler uses relative weights.
    w = w / w.mean()
    return w.astype(np.float64)


def train_padneutralized(model_save_path: str):
    _suppress_pil_decompression_bomb_warnings()

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    print("[*] Loading dataset (base loaders)...")
    train_loader_base, val_loader, test_loader, class_weights = get_data_loaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # Build a deterministic feature-extraction loader (no shuffle)
    print("[*] Extracting padding features on full TRAIN split...")
    feat_loader = DataLoader(
        train_loader_base.dataset,
        batch_size=256,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    X, y = _extract_padding_features(feat_loader)

    # Sanity: report baseline AUC of padding-only predictor
    clf_auc = LogisticRegression(max_iter=4000, class_weight="balanced").fit(X, y)
    auc_train = float(
        __import__("sklearn.metrics").metrics.roc_auc_score(y, clf_auc.predict_proba(X)[:, 1])
    )
    print(f"[*] TRAIN padding-only LR AUC (before weighting): {auc_train:.4f}")

    weights = _compute_propensity_weights(X, y)
    print(f"[*] Sampler weights: mean={weights.mean():.3f} min={weights.min():.3f} max={weights.max():.3f}")

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=len(weights),
        replacement=True,
    )
    train_loader = DataLoader(
        train_loader_base.dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
    )

    model = get_resnet18_grayscale().to(device)

    # Use original class imbalance weighting for loss
    class_weights = class_weights.to(device)
    pos_weight = torch.tensor([class_weights[1] / class_weights[0]]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    print(f"[*] Loss: BCEWithLogitsLoss (pos_weight: {pos_weight.item():.4f})")
    print(f"[*] Saving best model to: {model_save_path}")

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    start = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_total += labels.size(0)
            train_correct += ((logits > 0).float() == labels).sum().item()

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = 100.0 * train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)
                val_total += labels.size(0)
                val_correct += ((logits > 0).float() == labels).sum().item()

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100.0 * val_correct / val_total

        elapsed_min = (time.time() - start) / 60.0
        print(f"epoch={epoch+1}/{NUM_EPOCHS} elapsed_min={elapsed_min:.2f} "
              f"train_loss={epoch_train_loss:.4f} train_acc={epoch_train_acc:.2f} "
              f"val_loss={epoch_val_loss:.4f} val_acc={epoch_val_acc:.2f}")

        if epoch_val_loss < best_val_loss:
            torch.save(model.state_dict(), model_save_path)
            best_val_loss = epoch_val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            print("  save=1")
        else:
            epochs_no_improve += 1
            print(f"  save=0 no_improve={epochs_no_improve}/{EARLY_STOP_PATIENCE}")

        scheduler.step(epoch_val_loss)

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"early_stop epoch={epoch+1} best_epoch={best_epoch} best_val_loss={best_val_loss:.4f}")
            break

    # quick test accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images).squeeze(1)
            pred = (logits > 0).to(labels.dtype)
            correct += (pred == labels).sum().item()
            total += labels.numel()
    print(f"test_acc={correct/max(1,total):.4f} n={total}")


if __name__ == "__main__":
    out = os.path.join(str(LOGS_DIR), "resnet18_clean_padneutralized_sampler.pth")
    train_padneutralized(out)
