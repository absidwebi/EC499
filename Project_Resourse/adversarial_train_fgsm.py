"""
adversarial_train_fgsm.py
==========================
Adversarial training of 3C2D using FGSM attacks (Goodfellow et al. 2014).

Parallel to adversarial_train.py (PGD-based). Both scripts share:
  - Identical resume checkpoint format
  - Identical early stopping logic
  - Identical logging and curve generation
  - Identical evaluation protocol

Output:
  models/3c2d_malex_fgsm_adversarially_trained.pth  <- best robust weights
  models/at_3c2d_fgsm_full_checkpoint.pth            <- full resume state
  logs/adversarial_training_log_fgsm.txt
  logs/adversarial_training_curve_fgsm.png
  run_logs/adversarial_train_fgsm_YYYYMMDD_HHMMSS.log (created by tee)
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MALEX_DATASET_DIR_STR,
    MALEX_3C2D_CLEAN_MODEL_PATH_STR,
    MALEX_3C2D_FGSM_ADV_MODEL_PATH_STR,
    MODEL_OUTPUT_DIR,
    LOGS_DIR,
)
from dataset_loader import get_data_loaders
from models import MaleX3C2D

# ============================================================
# CONFIGURATION
# ============================================================
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
EARLY_STOP_PATIENCE = 5

# FGSM attack parameter
ADV_TRAIN_EPS = 0.05

# Resume control
RESUME_IF_CHECKPOINT_EXISTS = True

# Paths
CLEAN_MODEL_PATH = MALEX_3C2D_CLEAN_MODEL_PATH_STR
ROBUST_MODEL_PATH = MALEX_3C2D_FGSM_ADV_MODEL_PATH_STR
CHECKPOINT_PATH = str(MODEL_OUTPUT_DIR / "at_3c2d_fgsm_full_checkpoint.pth")
LOG_PATH = LOGS_DIR / "adversarial_training_log_fgsm.txt"
CURVE_PATH = str(LOGS_DIR / "adversarial_training_curve_fgsm.png")
# ============================================================

_criterion = nn.BCEWithLogitsLoss()


# ============================================================
# Attack
# ============================================================

def fgsm_attack(model, images, labels, eps):
    """Single-step FGSM used during adversarial training."""
    adv = images.clone().detach().requires_grad_(True)
    labels_f = labels.float().unsqueeze(1)
    loss = _criterion(model(adv), labels_f)
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        adv = torch.clamp(adv + eps * adv.grad.sign(), -1.0, 1.0)
    return adv.detach()


# ============================================================
# Checkpoint helpers (identical interface to adversarial_train.py)
# ============================================================

def save_full_checkpoint(
    epoch,
    model,
    optimizer,
    best_robust_val_acc,
    best_epoch,
    epochs_no_improve,
    log_lines,
    path,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_robust_val_acc": best_robust_val_acc,
            "best_epoch": best_epoch,
            "epochs_no_improve": epochs_no_improve,
            "log_lines": log_lines,
        },
        path,
    )


def load_full_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    print(
        f"[*] Resumed from checkpoint: epoch {ckpt['epoch'] + 1}, "
        f"best_robust_val_acc={ckpt['best_robust_val_acc']:.2f}% "
        f"(epoch {ckpt['best_epoch']})"
    )
    return {
        "start_epoch": ckpt["epoch"] + 1,
        "best_robust_val_acc": ckpt["best_robust_val_acc"],
        "best_epoch": ckpt["best_epoch"],
        "epochs_no_improve": ckpt["epochs_no_improve"],
        "log_lines": ckpt["log_lines"],
    }


# ============================================================
# Training curve
# ============================================================

def plot_curves(robust_accs, clean_accs, save_path, start_epoch=0):
    epochs = range(start_epoch + 1, start_epoch + len(robust_accs) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(
        list(epochs),
        robust_accs,
        "r-o",
        markersize=4,
        linewidth=2,
        label="Val Robust Accuracy (FGSM)",
    )
    plt.plot(
        list(epochs),
        clean_accs,
        "b-o",
        markersize=4,
        linewidth=2,
        label="Val Clean Accuracy",
    )
    plt.axhline(
        65,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label="Target robust lower bound (65%)",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(
        f"FGSM Adversarial Training - 3C2D on MaleX\n{chr(949)}={ADV_TRAIN_EPS}",
        fontsize=12,
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# Validation helpers
# ============================================================

def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.float().to(device)
            predicted = (model(images).squeeze(1) > 0).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def evaluate_robust_accuracy(model, loader, device):
    """Measure accuracy on FGSM-perturbed inputs."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        adv = fgsm_attack(model, images, labels, ADV_TRAIN_EPS)
        with torch.no_grad():
            predicted = (model(adv).squeeze(1) > 0).float()
            correct += (predicted == labels.float()).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


# ============================================================
# Main training loop
# ============================================================

def main():
    torch.manual_seed(42)
    print("=" * 65)
    print("  Stage 3 - FGSM Adversarial Training (3C2D on MaleX)")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"[*] Device        : {device}")
    print(f"[*] FGSM epsilon  : {ADV_TRAIN_EPS}")
    print(f"[*] Max epochs    : {NUM_EPOCHS}")
    print(f"[*] Early stop    : patience={EARLY_STOP_PATIENCE}")
    print(f"[*] Best weights  : {ROBUST_MODEL_PATH}")
    print(f"[*] Full ckpt     : {CHECKPOINT_PATH}")

    # Data
    print("[*] Loading MaleX dataset...")
    train_loader, val_loader, _, class_weights = get_data_loaders(
        data_dir=MALEX_DATASET_DIR_STR, batch_size=BATCH_SIZE, num_workers=0
    )
    class_weights = class_weights.to(device)

    # Model
    model = MaleX3C2D().to(device)
    print(f"[*] Parameters    : {sum(p.numel() for p in model.parameters()):,}")

    # Loss and Optimizer
    pos_weight = torch.tensor([class_weights[1] / class_weights[0]]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print(f"[*] pos_weight    : {pos_weight.item():.4f}")

    # Warm-start from clean model
    if not os.path.exists(CLEAN_MODEL_PATH):
        raise FileNotFoundError(
            f"Clean 3C2D model not found: {CLEAN_MODEL_PATH}\n"
            "Run train_3c2d.py first."
        )
    model.load_state_dict(torch.load(CLEAN_MODEL_PATH, map_location=device))
    print(f"[*] Warm-started from: {CLEAN_MODEL_PATH}")

    # Resume or fresh start
    start_epoch = 0
    best_robust_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    log_lines = ["Epoch | Train Loss | Train Acc | Val Clean | Val Robust (FGSM)\n"]
    robust_accs = []
    clean_accs = []

    if RESUME_IF_CHECKPOINT_EXISTS and os.path.exists(CHECKPOINT_PATH):
        state = load_full_checkpoint(CHECKPOINT_PATH, model, optimizer, device)
        start_epoch = state["start_epoch"]
        best_robust_val_acc = state["best_robust_val_acc"]
        best_epoch = state["best_epoch"]
        epochs_no_improve = state["epochs_no_improve"]
        log_lines = state["log_lines"]
        print(f"[*] Resuming from epoch {start_epoch + 1}")
    else:
        print("[*] Starting fresh FGSM adversarial training run.")
    print("-" * 65)

    start_time = time.time()

    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_start = time.time()

        # Adversarial train
        model.train()
        train_loss = train_correct = train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Generate FGSM adversarial examples (eval mode for stable BN)
            model.eval()
            adv_images = fgsm_attack(model, images, labels, ADV_TRAIN_EPS)

            # Train on adversarial examples
            model.train()
            optimizer.zero_grad()
            labels_f = labels.float().unsqueeze(1)
            outputs = model(adv_images)
            loss = criterion(outputs, labels_f)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            predicted = (outputs > 0).float()
            train_total += labels_f.size(0)
            train_correct += (predicted == labels_f).sum().item()

            if (batch_idx + 1) % 50 == 0:
                avg_t = (time.time() - epoch_start) / (batch_idx + 1)
                est_rem = avg_t * (len(train_loader) - (batch_idx + 1))
                running_acc = 100.0 * train_correct / train_total
                print(
                    f"  Epoch [{epoch+1}/{NUM_EPOCHS}] "
                    f"Batch [{batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}  "
                    f"Acc: {running_acc:.2f}%  "
                    f"Est: {est_rem/60:.1f}m rem"
                )

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = 100.0 * train_correct / train_total

        # Validation
        clean_val_acc = evaluate_accuracy(model, val_loader, device)
        robust_val_acc = evaluate_robust_accuracy(model, val_loader, device)

        robust_accs.append(robust_val_acc)
        clean_accs.append(clean_val_acc)

        elapsed = (time.time() - start_time) / 60
        log_line = (
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Train Acc: {epoch_train_acc:.2f}% | "
            f"Val Clean: {clean_val_acc:.2f}% | "
            f"Val Robust (FGSM): {robust_val_acc:.2f}% | "
            f"Elapsed: {elapsed:.1f}min"
        )
        print(f"\n{'=' * 65}")
        print(f"  {log_line}")
        log_lines.append(log_line + "\n")

        # Checkpoint logic
        if robust_val_acc > best_robust_val_acc:
            improvement = robust_val_acc - best_robust_val_acc
            print(f"  * Robust val improved by {improvement:.2f}% -> saving best weights")
            torch.save(model.state_dict(), ROBUST_MODEL_PATH)
            best_robust_val_acc = robust_val_acc
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(
                f"  No improvement for {epochs_no_improve}/{EARLY_STOP_PATIENCE} epochs "
                f"(best: {best_robust_val_acc:.2f}% at epoch {best_epoch})"
            )

        # Save full checkpoint after every epoch
        save_full_checkpoint(
            epoch,
            model,
            optimizer,
            best_robust_val_acc,
            best_epoch,
            epochs_no_improve,
            log_lines,
            CHECKPOINT_PATH,
        )
        print(f"  Full checkpoint saved (epoch {epoch+1})")

        # Save curve after every epoch
        plot_curves(robust_accs, clean_accs, CURVE_PATH, start_epoch=start_epoch)

        # Write log after every epoch
        with open(LOG_PATH, "w") as f:
            f.writelines(log_lines)

        # Early stopping
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(
                f"\nEarly stopping at epoch {epoch+1}. "
                f"Best robust val acc: {best_robust_val_acc:.2f}% (epoch {best_epoch})."
            )
            break

        print(f"{'=' * 65}\n")

    total_time = time.time() - start_time
    print(f"\nFGSM Adversarial Training complete in {total_time/60:.2f} minutes")
    print(f"   Best robust val acc (FGSM) : {best_robust_val_acc:.2f}% (epoch {best_epoch})")
    print(f"   Best weights saved         : {ROBUST_MODEL_PATH}")
    print(f"   Full checkpoint            : {CHECKPOINT_PATH}")
    print(f"   Training curve             : {CURVE_PATH}")
    print(f"   Log file                   : {LOG_PATH}")
    print("\nNext step: run evaluate_attacks_fixed.py --model 3c2d_fgsm")


if __name__ == "__main__":
    main()
