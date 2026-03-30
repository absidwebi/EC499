"""
Train the MaleX3C2D model (Mohammed et al. 2021 architecture - FIXED VERSION).
Supports resume from checkpoint for long training runs.
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
from models import MaleX3C2D
from config import MALEX_DATASET_DIR_STR, MALEX_3C2D_CLEAN_MODEL_PATH_STR, LOGS_DIR

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR             = MALEX_DATASET_DIR_STR
BEST_MODEL_PATH      = MALEX_3C2D_CLEAN_MODEL_PATH_STR

# Full checkpoint path (saves optimizer + scheduler state for resume)
CHECKPOINT_DIR       = os.path.join(os.path.dirname(BEST_MODEL_PATH))
FULL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "3c2d_malex_full_checkpoint.pth")

BATCH_SIZE           = 64
NUM_EPOCHS           = 70
LEARNING_RATE        = 1e-3
WEIGHT_DECAY         = 1e-4
EARLY_STOP_PATIENCE  = 10   # Higher patience for longer run

# Set to True to resume from FULL_CHECKPOINT_PATH if it exists.
# Set to False to always start fresh (will overwrite existing checkpoints).
RESUME_IF_CHECKPOINT_EXISTS = True
# ============================================================


def save_full_checkpoint(epoch, model, optimizer, scheduler,
                         best_val_loss, best_epoch, path):
    """Save everything needed to resume training exactly where it stopped."""
    torch.save({
        'epoch'         : epoch,
        'model_state'   : model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_val_loss' : best_val_loss,
        'best_epoch'    : best_epoch,
    }, path)


def load_full_checkpoint(path, model, optimizer, scheduler, device):
    """Load a full checkpoint. Returns (start_epoch, best_val_loss, best_epoch)."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    print(f"[*] Resumed from checkpoint: epoch {ckpt['epoch']+1}, "
          f"best_val_loss={ckpt['best_val_loss']:.4f} (epoch {ckpt['best_epoch']})")
    return ckpt['epoch'] + 1, ckpt['best_val_loss'], ckpt['best_epoch']


def plot_curves(train_accs, val_accs, save_path, start_epoch=0):
    epochs = range(start_epoch + 1, start_epoch + len(train_accs) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(list(epochs), train_accs, 'b-o', markersize=4,
             label='Train Accuracy', linewidth=2)
    plt.plot(list(epochs), val_accs,   'r-o', markersize=4,
             label='Val Accuracy',   linewidth=2)
    plt.title('3C2D Model (Fixed) - MaleX Training vs Validation Accuracy', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train():
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    print("=" * 60)
    print("  3C2D Model Training - FIXED VERSION (2-4M params)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    # -- Data ------------------------------------------------
    print("[*] Loading MaleX dataset...")
    train_loader, val_loader, _, class_weights = get_data_loaders(
        data_dir=DATA_DIR, batch_size=BATCH_SIZE, num_workers=0)
    class_weights = class_weights.to(device)

    # -- Model -----------------------------------------------
    model = MaleX3C2D().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Model: MaleX3C2D | Parameters: {total_params:,}")
    if total_params > 10_000_000:
        raise RuntimeError(
            f"Model has {total_params:,} parameters - architecture bug still present. "
            "Check that models.py was saved correctly after Task 1.")

    # -- Loss, Optimizer, Scheduler --------------------------
    pos_weight = torch.tensor([class_weights[1] / class_weights[0]]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = optim.Adam(model.parameters(),
                            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # CosineAnnealingLR: smoothly reduces LR from LEARNING_RATE to near-zero
    # over NUM_EPOCHS. Faster convergence than ReduceLROnPlateau for longer runs.
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    print(f"[*] Optimizer  : Adam (LR={LEARNING_RATE}, WD={WEIGHT_DECAY})")
    print(f"[*] Scheduler  : CosineAnnealingLR (T_max={NUM_EPOCHS}, eta_min=1e-6)")
    print(f"[*] Loss       : BCEWithLogitsLoss (pos_weight={pos_weight.item():.4f})")
    print(f"[*] Batch size : {BATCH_SIZE}  |  Max epochs : {NUM_EPOCHS}")
    print(f"[*] Early stop : patience={EARLY_STOP_PATIENCE}")

    # -- Resume or Fresh Start -------------------------------
    start_epoch       = 0
    best_val_loss     = float('inf')
    best_epoch        = 0
    epochs_no_improve = 0
    train_accs, val_accs = [], []

    if RESUME_IF_CHECKPOINT_EXISTS and os.path.exists(FULL_CHECKPOINT_PATH):
        start_epoch, best_val_loss, best_epoch = load_full_checkpoint(
            FULL_CHECKPOINT_PATH, model, optimizer, scheduler, device)
        # Advance scheduler to match resumed state
        # (already restored by load_full_checkpoint via scheduler.state_dict())
    else:
        print("[*] Starting fresh training run.")

    print("-" * 60)

    curve_path = os.path.join(str(LOGS_DIR), "training_curve_3c2d_fixed.png")
    start_time = time.time()

    for epoch in range(start_epoch, NUM_EPOCHS):
        # -- TRAIN -------------------------------------------
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

            if (batch_idx + 1) % 100 == 0:
                running_acc = 100.0 * train_correct / train_total
                current_lr  = optimizer.param_groups[0]['lr']
                print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}] "
                      f"Step [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}  "
                      f"Acc: {running_acc:.2f}%  "
                      f"LR: {current_lr:.2e}")

        epoch_train_loss = train_loss / train_total
        epoch_train_acc  = 100.0 * train_correct / train_total

        # -- VALIDATE ----------------------------------------
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

        # Step scheduler (cosine - call once per epoch)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # -- Epoch Summary -----------------------------------
        elapsed = (time.time() - start_time) / 60
        print(f"\n{'='*60}")
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS}  ({elapsed:.1f} min elapsed)")
        print(f"  Train  - Loss: {epoch_train_loss:.4f}  Acc: {epoch_train_acc:.2f}%")
        print(f"  Val    - Loss: {epoch_val_loss:.4f}  Acc: {epoch_val_acc:.2f}%")
        print(f"  LR now: {current_lr:.2e}")

        # -- Checkpoint Logic --------------------------------
        if epoch_val_loss < best_val_loss:
            improvement = best_val_loss - epoch_val_loss
            best_val_loss = epoch_val_loss
            best_epoch    = epoch + 1
            epochs_no_improve = 0

            # Save best model weights (for inference)
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  New best val loss (improved {improvement:.4f}) -> saved best weights")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{EARLY_STOP_PATIENCE} epochs")

        # Always save full checkpoint after every epoch (for resume)
        save_full_checkpoint(epoch, model, optimizer, scheduler,
                             best_val_loss, best_epoch, FULL_CHECKPOINT_PATH)
        print(f"  Full checkpoint saved (epoch {epoch+1})")

        # -- Early Stopping ----------------------------------
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}. "
                  f"Best was epoch {best_epoch}.")
            break

        print(f"{'='*60}\n")

    # -- Post-Training --------------------------------------
    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/60:.2f} minutes")
    print(f"   Best val loss : {best_val_loss:.4f}  (epoch {best_epoch})")
    print(f"   Best weights  : {BEST_MODEL_PATH}")
    print(f"   Full ckpt     : {FULL_CHECKPOINT_PATH}")
    plot_curves(train_accs, val_accs, curve_path, start_epoch=start_epoch)
    print(f"   Curve saved   : {curve_path}")


if __name__ == "__main__":
    train()
