"""
train_efficientnet.py — Stage 2 (EfficientNet-B0 Comparison)
=============================================================
Trains EfficientNet-B0 on the same dataset as ResNet-18 using
identical hyperparameters for a controlled architectural comparison.

Scientific purpose:
    If both ResNet-18 and EfficientNet-B0 show similar logit separation
    and evasion rates, this confirms the behaviour is dataset-level
    (source bias) rather than architecture-specific.

Output: efficientnet_b0_clean_vulnerable.pth
        logs/efficientnet_training_curve.png
        logs/train_efficientnet.log
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset_loader import get_data_loaders
from models import get_efficientnet_b0_grayscale
from config import MALIMG_ARCHIVE_DIR_STR, EFFICIENTNET_CLEAN_MODEL_PATH_STR, LOGS_DIR

# === CONFIGURATION — identical to train.py for fair comparison ===
DATA_DIR        = MALIMG_ARCHIVE_DIR_STR
MODEL_SAVE_PATH = EFFICIENTNET_CLEAN_MODEL_PATH_STR
BATCH_SIZE          = 32
NUM_EPOCHS          = 20
LEARNING_RATE       = 1e-4
EARLY_STOP_PATIENCE = 5
LOG_PATH            = str(LOGS_DIR / "train_efficientnet.log")
# ================================================================


def plot_training_curves(train_accs, val_accs, save_path):
    epochs = range(1, len(train_accs) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accs, 'b-o', label='Train Accuracy', linewidth=2)
    plt.plot(epochs, val_accs,   'r-o', label='Val Accuracy',   linewidth=2)
    plt.title('Training vs Validation Accuracy — EfficientNet-B0 Clean Baseline', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[*] Training curve saved to: {save_path}")


def train():
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    print("🚀 Initializing Stage 2: EfficientNet-B0 Clean Baseline Training")
    print("-" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    print("[*] Loading dataset...")
    train_loader, val_loader, _, class_weights = get_data_loaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=0
    )
    class_weights = class_weights.to(device)

    print("[*] Initializing EfficientNet-B0 (Grayscale 1-channel)...")
    model = get_efficientnet_b0_grayscale().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Total parameters: {total_params:,}")

    pos_weight = torch.tensor([class_weights[1] / class_weights[0]]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    print(f"[*] Optimizer : AdamW  (LR: {LEARNING_RATE})")
    print(f"[*] Scheduler : ReduceLROnPlateau (factor=0.5, patience=3)")
    print(f"[*] Loss      : BCEWithLogitsLoss (pos_weight: {pos_weight.item():.4f})")
    print(f"[*] Epochs    : {NUM_EPOCHS}  |  Early stop patience: {EARLY_STOP_PATIENCE}")
    print("-" * 60)

    best_val_loss     = float('inf')
    best_epoch        = 0
    epochs_no_improve = 0
    train_accs        = []
    val_accs          = []
    log_lines         = []
    curve_save_path   = str(LOGS_DIR / "efficientnet_training_curve.png")
    start_time        = time.time()

    for epoch in range(NUM_EPOCHS):
        # ── TRAIN ──────────────────────────────────────────────
        model.train()
        train_loss = train_correct = train_total = 0

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

        # ── VALIDATION ─────────────────────────────────────────
        model.eval()
        val_loss = val_correct = val_total = 0
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

        elapsed    = (time.time() - start_time) / 60
        current_lr = optimizer.param_groups[0]['lr']

        summary = (f"Epoch {epoch+1}/{NUM_EPOCHS} | "
                   f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
                   f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}% | "
                   f"LR: {current_lr:.2e} | {elapsed:.1f}min")
        print(f"\n{'='*60}")
        print(f"  {summary}")
        log_lines.append(summary + "\n")

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

        scheduler.step(epoch_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            print(f"  📉 LR reduced: {current_lr:.2e} → {new_lr:.2e}")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n🛑 Early stopping at epoch {epoch+1}. Best was epoch {best_epoch}.")
            break

        print(f"{'='*60}\n")

    total_time = time.time() - start_time
    print(f"\n🎉 EfficientNet-B0 training complete in {total_time/60:.2f} minutes")
    print(f"   Best val loss : {best_val_loss:.4f}  (epoch {best_epoch})")
    print(f"   Weights saved : {MODEL_SAVE_PATH}")

    plot_training_curves(train_accs, val_accs, curve_save_path)

    with open(LOG_PATH, "w") as f:
        f.writelines(log_lines)
    print(f"   Log saved     : {LOG_PATH}")
    print("\nReady for adversarial training: run adversarial_train_efficientnet.py")


if __name__ == "__main__":
    train()
