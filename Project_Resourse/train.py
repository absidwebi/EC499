import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — safe for headless servers
import matplotlib.pyplot as plt

from dataset_loader import get_data_loaders
from models import get_resnet18_grayscale
from config import MALIMG_ARCHIVE_DIR_STR, RESNET_CLEAN_MODEL_PATH_STR, LOGS_DIR

# === CONFIGURATION ===
DATA_DIR        = os.environ.get("DATA_DIR_OVERRIDE", MALIMG_ARCHIVE_DIR_STR)
MODEL_SAVE_PATH = RESNET_CLEAN_MODEL_PATH_STR

BATCH_SIZE    = 32
NUM_EPOCHS    = 20
LEARNING_RATE = 1e-4
EARLY_STOP_PATIENCE = 5   # Stop if val loss does not improve for this many epochs
# Optional: neutralize padding-layout shortcut by sampling.
# Enables a WeightedRandomSampler where weights are computed as inverse
# propensity from (pad_rows_all_-1, pad_cols_all_-1, frac_-1_pixels).
PAD_NEUTRALIZE_SAMPLER = os.environ.get("PAD_NEUTRALIZE_SAMPLER", "0") == "1"
PAD_NEUTRALIZE_WEIGHT_CLIP_MAX = float(os.environ.get("PAD_NEUTRALIZE_WEIGHT_CLIP_MAX", "20.0"))
# =====================


def _suppress_pil_decompression_bomb_warnings():
    # Some Malimg-derived images can be extremely large on disk. We never resize
    # or interpolate them; we center-crop/pad to 256 in dataset_loader. The
    # warning is noisy in logs and does not indicate corrupt decoding here.
    try:
        from PIL import Image
        import warnings

        warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
    except Exception:
        pass


def plot_training_curves(train_accs, val_accs, save_path):
    """Save a training vs validation accuracy curve to disk."""
    epochs = range(1, len(train_accs) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accs, 'b-o', label='Train Accuracy', linewidth=2)
    plt.plot(epochs, val_accs,   'r-o', label='Val Accuracy',   linewidth=2)
    plt.title('Training vs Validation Accuracy — ResNet-18 Clean Baseline',
              fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[*] Training curve saved to: {save_path}")


def train():
    _suppress_pil_decompression_bomb_warnings()

    # Reproducibility
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    print("🚀 Initializing Stage 2: ResNet-18 Clean Baseline Training")
    print("-" * 60)

    # 1. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    # 2. DataLoaders
    # num_workers=0 prevents CUDA fork OOM on Linux
    print("[*] Loading dataset...")
    train_loader, val_loader, _, class_weights = get_data_loaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=0
    )
    class_weights = class_weights.to(device)

    if PAD_NEUTRALIZE_SAMPLER:
        # Compute padding-feature weights on the training dataset.
        # We reuse the same transformed tensors to stay faithful to the model's view.
        from torch.utils.data import DataLoader as _DL
        from torch.utils.data import WeightedRandomSampler

        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score
        except Exception as e:
            raise RuntimeError(
                "PAD_NEUTRALIZE_SAMPLER=1 requires scikit-learn. "
                f"Import failed: {e}"
            )

        print("[*] PAD_NEUTRALIZE_SAMPLER=1: extracting padding features on TRAIN...")
        feat_loader = _DL(
            train_loader.dataset,
            batch_size=256,
            shuffle=False,
            num_workers=0,
        )
        X = []
        y = []
        with torch.no_grad():
            for imgs, lbls in feat_loader:
                x = imgs.squeeze(1)
                is_neg1 = (x == -1.0)
                pad_rows = is_neg1.all(dim=2).sum(dim=1).cpu().numpy()
                pad_cols = is_neg1.all(dim=1).sum(dim=1).cpu().numpy()
                frac = is_neg1.float().mean(dim=(1, 2)).cpu().numpy()
                X.append(__import__("numpy").stack([pad_rows, pad_cols, frac], axis=1))
                y.append(lbls.cpu().numpy())
        X = __import__("numpy").concatenate(X, axis=0)
        y = __import__("numpy").concatenate(y, axis=0).astype(int)

        clf = LogisticRegression(max_iter=4000, class_weight="balanced")
        clf.fit(X, y)
        p = clf.predict_proba(X)
        auc = roc_auc_score(y, p[:, 1])
        p_true = p[__import__("numpy").arange(len(y)), y]
        w = 1.0 / __import__("numpy").maximum(p_true, 1e-6)
        w = __import__("numpy").minimum(w, PAD_NEUTRALIZE_WEIGHT_CLIP_MAX)
        w = w / w.mean()
        print(f"[*] TRAIN padding-only LR AUC (for sampler weights): {auc:.4f}")
        print(f"[*] Sampler weights: mean={w.mean():.3f} min={w.min():.3f} max={w.max():.3f}")

        sampler = WeightedRandomSampler(
            weights=w.tolist(),
            num_samples=len(w),
            replacement=True,
        )
        train_loader = _DL(
            train_loader.dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=0,
        )
        print("[*] Using WeightedRandomSampler for training")

    # 3. Model
    print("[*] Initializing ResNet-18 (Grayscale 1-channel)...")
    model = get_resnet18_grayscale().to(device)

    # 4. Loss, Optimizer, Scheduler
    pos_weight = torch.tensor([class_weights[1] / class_weights[0]]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # ReduceLROnPlateau: halve LR if val loss does not improve for 3 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )

    print(f"[*] Optimizer : AdamW  (LR: {LEARNING_RATE})")
    print(f"[*] Scheduler : ReduceLROnPlateau (factor=0.5, patience=3)")
    print(f"[*] Loss      : BCEWithLogitsLoss (pos_weight: {pos_weight.item():.4f})")
    print(f"[*] Epochs    : {NUM_EPOCHS}  |  Early stop patience: {EARLY_STOP_PATIENCE}")
    print("-" * 60)

    # 5. Training State
    best_val_loss      = float('inf')
    best_epoch         = 0
    epochs_no_improve  = 0
    train_accs         = []
    val_accs           = []
    curve_save_path    = os.path.join(str(LOGS_DIR), "training_curve.png")

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        # ── TRAIN ──────────────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * images.size(0)
            predicted      = (outputs > 0).float()
            train_total   += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            if (batch_idx + 1) % 50 == 0:
                running_acc = 100.0 * train_correct / train_total
                print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}] "
                      f"Step [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}  "
                      f"Running Acc: {running_acc:.2f}%")

        epoch_train_loss = train_loss / train_total
        epoch_train_acc  = 100.0 * train_correct / train_total

        # ── VALIDATION ─────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                outputs    = model(images)
                loss       = criterion(outputs, labels)

                val_loss    += loss.item() * images.size(0)
                predicted    = (outputs > 0).float()
                val_total   += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / val_total
        epoch_val_acc  = 100.0 * val_correct / val_total

        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)

        # ── Epoch Summary ───────────────────────────────────────────────
        elapsed = (time.time() - start_time) / 60
        print(f"\n{'='*60}")
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS}  ({elapsed:.1f} min elapsed)")
        print(f"  Train  — Loss: {epoch_train_loss:.4f}  Acc: {epoch_train_acc:.2f}%")
        print(f"  Val    — Loss: {epoch_val_loss:.4f}  Acc: {epoch_val_acc:.2f}%")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  LR now: {current_lr:.2e}")

        # ── Checkpoint ──────────────────────────────────────────────────
        if epoch_val_loss < best_val_loss:
            improvement = best_val_loss - epoch_val_loss
            print(f"  ⭐ Val loss improved by {improvement:.4f} → saving model")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            best_val_loss     = epoch_val_loss
            best_epoch        = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{EARLY_STOP_PATIENCE} epochs")

        # ── LR Scheduler ────────────────────────────────────────────────
        scheduler.step(epoch_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            print(f"  📉 LR reduced: {current_lr:.2e} → {new_lr:.2e}")

        # ── Early Stopping ──────────────────────────────────────────────
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n🛑 Early stopping triggered at epoch {epoch+1}. "
                  f"Best was epoch {best_epoch}.")
            break

        print(f"{'='*60}\n")

    # ── Post-Training ───────────────────────────────────────────────────
    total_time = time.time() - start_time
    print(f"\n🎉 Training complete in {total_time/60:.2f} minutes")
    print(f"   Best val loss : {best_val_loss:.4f}  (epoch {best_epoch})")
    print(f"   Weights saved : {MODEL_SAVE_PATH}")

    plot_training_curves(train_accs, val_accs, curve_save_path)
    print("\nReady for Stage 3: Adversarial Attack Vulnerability Testing.")


if __name__ == "__main__":
    train()
