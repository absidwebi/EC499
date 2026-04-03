"""
Stage 3 — Part 2: Adversarial Training Defense
================================================
Implements Madry et al. (2018) adversarial training to harden the CNN.

Core Idea:
    During each training iteration, instead of training on the clean input x,
    we first generate a PGD adversarial example x_adv from x, then train
    the model to correctly classify x_adv. This forces the model to learn
    decision boundaries robust to worst-case perturbations.

Output:
    Saves the robust model to models/custom_cnn_adversarially_trained.pth
    for comparison against the clean baseline in final evaluation.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    MALEX_DATASET_DIR_STR,
    MALEX_3C2D_CLEAN_MODEL_PATH_STR,
    MALEX_3C2D_ADV_MODEL_PATH_STR,
    RESNET_MALEX_CLEAN_MODEL_PATH_STR,
    RESNET_MALEX_ADV_MODEL_PATH_STR,
    RESNET_MALEX_PRETRAINED_CLEAN_PATH_STR,
    LOGS_DIR,
)
from dataset_loader import get_data_loaders
from models import MaleX3C2D, get_resnet18_grayscale, get_resnet18_pretrained_grayscale

# ============================================================
# CONFIGURATION
# ============================================================
BATCH_SIZE          = 16
NUM_EPOCHS          = 50      # Continue from epoch 35 checkpoint to epoch 50
LEARNING_RATE       = 1e-4
EARLY_STOP_PATIENCE = 5       # Stop if robust val acc does not improve

# PGD attack parameters used DURING adversarial training
ADV_TRAIN_EPS   = 0.05
ADV_TRAIN_ALPHA = 0.01
ADV_TRAIN_STEPS = 7

# Model options: "3c2d", "resnet", "resnet_pretrained"
MODEL_VARIANT = "3c2d"

# Resume control - set False to force a fresh start
RESUME_IF_CHECKPOINT_EXISTS = True
# ============================================================

# Shared loss criterion for attack gradient computation
_criterion = nn.BCEWithLogitsLoss()


def pgd_attack(model, images, labels, eps, alpha, steps):
    """Manual PGD attack compatible with BCEWithLogitsLoss (Madry et al., 2018)."""
    labels_f    = labels.float().unsqueeze(1)
    adv_images  = images.clone().detach()
    adv_images  = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
    adv_images  = torch.clamp(adv_images, -1.0, 1.0).detach()

    for _ in range(steps):
        adv_images.requires_grad_(True)
        outputs = model(adv_images)
        loss    = _criterion(outputs, labels_f)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            adv_images = adv_images + alpha * adv_images.grad.sign()
            delta      = torch.clamp(adv_images - images, -eps, eps)
            adv_images = torch.clamp(images + delta, -1.0, 1.0).detach()
    return adv_images


def save_full_checkpoint(epoch, model, optimizer, best_robust_val_acc,
                         best_epoch, epochs_no_improve, log_lines, path):
    """Save full training state for resume capability."""
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'best_robust_val_acc': best_robust_val_acc,
        'best_epoch': best_epoch,
        'epochs_no_improve': epochs_no_improve,
        'log_lines': log_lines,
    }, path)


def load_full_checkpoint(path, model, optimizer, device):
    """Load checkpoint and return training state dict."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    print(f"[*] Resumed from checkpoint: epoch {ckpt['epoch'] + 1}, "
          f"best_robust_val_acc={ckpt['best_robust_val_acc']:.2f}% "
          f"(epoch {ckpt['best_epoch']})")
    return {
        'start_epoch': ckpt['epoch'] + 1,
        'best_robust_val_acc': ckpt['best_robust_val_acc'],
        'best_epoch': ckpt['best_epoch'],
        'epochs_no_improve': ckpt['epochs_no_improve'],
        'log_lines': ckpt['log_lines'],
    }


def plot_training_curves(robust_accs, clean_accs, save_path, start_epoch=0):
    """Save adversarial training curve (robust + clean val accuracy)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = range(start_epoch + 1, start_epoch + len(robust_accs) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(list(epochs), robust_accs, 'r-o', markersize=4, linewidth=2,
             label='Val Robust Accuracy (PGD)')
    plt.plot(list(epochs), clean_accs, 'b-o', markersize=4, linewidth=2,
             label='Val Clean Accuracy')
    plt.axhline(65, color='gray', linestyle='--', linewidth=1, alpha=0.6,
                label='Target robust lower bound (65%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Adversarial Training - {MODEL_VARIANT.upper()} on MaleX\n'
              f'ε={ADV_TRAIN_EPS}, α={ADV_TRAIN_ALPHA}, steps={ADV_TRAIN_STEPS}',
              fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# =====================


def get_model_bundle(model_variant):
    variant = model_variant.lower().strip()
    if variant == "3c2d":
        return (
            "MaleX3C2D",
            MaleX3C2D(),
            MALEX_3C2D_CLEAN_MODEL_PATH_STR,
            MALEX_3C2D_ADV_MODEL_PATH_STR,
        )
    if variant == "resnet":
        return (
            "ResNet-18",
            get_resnet18_grayscale(),
            RESNET_MALEX_CLEAN_MODEL_PATH_STR,
            RESNET_MALEX_ADV_MODEL_PATH_STR,
        )
    if variant == "resnet_pretrained":
        return (
            "ResNet-18 Pretrained",
            get_resnet18_pretrained_grayscale(),
            RESNET_MALEX_PRETRAINED_CLEAN_PATH_STR,
            RESNET_MALEX_ADV_MODEL_PATH_STR,
        )
    raise ValueError(
        f"Unsupported MODEL_VARIANT='{model_variant}'. "
        "Use one of: '3c2d', 'resnet', 'resnet_pretrained'."
    )


def evaluate_accuracy(model, loader, device):
    """Measure Accuracy on a DataLoader, returns percentage float."""
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.float().to(device)
            outputs   = model(images)
            predicted = (outputs > 0).float().squeeze(1)
            correct  += (predicted == labels).sum().item()
            total    += labels.size(0)
    return (correct / total) * 100.0


def evaluate_robust_accuracy_manual(model, loader, device):
    """Measure Accuracy on adversarially perturbed inputs using manual PGD."""
    model.eval()
    correct = 0
    total   = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = pgd_attack(model, images, labels,
                                eps=ADV_TRAIN_EPS, alpha=ADV_TRAIN_ALPHA, steps=20)
        with torch.no_grad():
            outputs   = model(adv_images)
            predicted = (outputs > 0).float().squeeze(1)
            correct  += (predicted == labels.float()).sum().item()
            total    += labels.size(0)
    return (correct / total) * 100.0


def main():
    torch.manual_seed(42)
    print("=" * 60)
    print(f"  Stage 3 - Part 2: Adversarial Training ({MODEL_VARIANT.upper()})")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"[*] Device: {device}")
    print(f"[*] Model variant: {MODEL_VARIANT}")
    print(f"[*] PGD: ε={ADV_TRAIN_EPS}, α={ADV_TRAIN_ALPHA}, steps={ADV_TRAIN_STEPS}")
    print(f"[*] Max epochs: {NUM_EPOCHS}  |  Early stop patience: {EARLY_STOP_PATIENCE}")

    # -- Data ---------------------------------------------------------
    train_loader, val_loader, _, class_weights = get_data_loaders(
        data_dir=MALEX_DATASET_DIR_STR, batch_size=BATCH_SIZE, num_workers=0
    )
    class_weights = class_weights.to(device)

    # -- Model, Loss, Optimizer ---------------------------------------
    model_name, model, clean_model_path, robust_model_save_path = get_model_bundle(MODEL_VARIANT)
    model = model.to(device)

    print(f"[*] Warm-starting {model_name} from: {clean_model_path}")
    if not os.path.exists(clean_model_path):
        raise FileNotFoundError(f"Clean model not found: {clean_model_path}")
    model.load_state_dict(torch.load(clean_model_path, map_location=device))

    pos_weight = torch.tensor([class_weights[1] / class_weights[0]]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"[*] pos_weight: {pos_weight.item():.4f}")
    print(f"[*] Best model path: {robust_model_save_path}")

    # -- Checkpoint paths ---------------------------------------------
    model_tag = MODEL_VARIANT.lower().replace(" ", "_")
    checkpoint_dir = os.path.dirname(robust_model_save_path)
    full_checkpoint_path = os.path.join(checkpoint_dir,
                                        f"at_{model_tag}_full_checkpoint.pth")
    log_path = LOGS_DIR / f"adversarial_training_log_{model_tag}.txt"
    curve_path = str(LOGS_DIR / f"adversarial_training_curve_{model_tag}.png")

    # -- Resume or fresh start ----------------------------------------
    start_epoch = 0
    best_robust_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    log_lines = ["Epoch | Train Loss | Train Acc | Val Clean | Val Robust\n"]
    robust_accs = []
    clean_accs = []

    if RESUME_IF_CHECKPOINT_EXISTS and os.path.exists(full_checkpoint_path):
        state = load_full_checkpoint(full_checkpoint_path, model, optimizer, device)
        start_epoch = state['start_epoch']
        best_robust_val_acc = state['best_robust_val_acc']
        best_epoch = state['best_epoch']
        epochs_no_improve = state['epochs_no_improve']
        log_lines = state['log_lines']
        print(f"[*] Resuming from epoch {start_epoch + 1}")
    else:
        print("[*] Starting fresh adversarial training run.")
    print("-" * 60)

    start_time = time.time()

    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_start = time.time()

        # -- ADVERSARIAL TRAIN ---------------------------------------
        model.train()
        train_loss = train_correct = train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Generate adversarial examples (eval mode for stable BN stats)
            model.eval()
            adv_images = pgd_attack(model, images, labels,
                                    eps=ADV_TRAIN_EPS,
                                    alpha=ADV_TRAIN_ALPHA,
                                    steps=ADV_TRAIN_STEPS)

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
                print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}] "
                      f"Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}  "
                      f"Acc: {running_acc:.2f}%  "
                      f"Est: {est_rem/60:.1f}m rem")

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = 100.0 * train_correct / train_total

        # -- VALIDATION -----------------------------------------------
        clean_val_acc = evaluate_accuracy(model, val_loader, device)
        robust_val_acc = evaluate_robust_accuracy_manual(model, val_loader, device)

        robust_accs.append(robust_val_acc)
        clean_accs.append(clean_val_acc)

        elapsed = (time.time() - start_time) / 60
        log_line = (f"Epoch {epoch+1}/{NUM_EPOCHS} | "
                    f"Train Loss: {epoch_train_loss:.4f} | "
                    f"Train Acc: {epoch_train_acc:.2f}% | "
                    f"Val Clean: {clean_val_acc:.2f}% | "
                    f"Val Robust: {robust_val_acc:.2f}% | "
                    f"Elapsed: {elapsed:.1f}min")
        print(f"\n{'='*60}")
        print(f"  {log_line}")
        log_lines.append(log_line + "\n")

        # -- Checkpoint Logic -----------------------------------------
        if robust_val_acc > best_robust_val_acc:
            improvement = robust_val_acc - best_robust_val_acc
            print(f"  ⭐ Robust val improved by {improvement:.2f}% -> saving best weights")
            torch.save(model.state_dict(), robust_model_save_path)
            best_robust_val_acc = robust_val_acc
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{EARLY_STOP_PATIENCE} epochs "
                  f"(best: {best_robust_val_acc:.2f}% at epoch {best_epoch})")

        # Save full checkpoint after every epoch (for safe resume)
        save_full_checkpoint(
            epoch, model, optimizer,
            best_robust_val_acc, best_epoch, epochs_no_improve,
            log_lines, full_checkpoint_path
        )
        print(f"  Full checkpoint saved (epoch {epoch+1})")

        # Save training curve after every epoch (overwrite)
        plot_training_curves(robust_accs, clean_accs, curve_path,
                             start_epoch=start_epoch)

        # Write log file after every epoch (overwrite with cumulative content)
        with open(log_path, "w") as f:
            f.writelines(log_lines)

        # -- Early Stopping -------------------------------------------
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}. "
                  f"Best robust val acc: {best_robust_val_acc:.2f}% (epoch {best_epoch}).")
            break

        print(f"{'='*60}\n")

    # -- Post-Training -------------------------------------------------
    total_time = time.time() - start_time
    print(f"\nAdversarial Training complete in {total_time/60:.2f} minutes")
    print(f"   Best robust val acc : {best_robust_val_acc:.2f}% (epoch {best_epoch})")
    print(f"   Best weights saved  : {robust_model_save_path}")
    print(f"   Full checkpoint     : {full_checkpoint_path}")
    print(f"   Training curve      : {curve_path}")
    print(f"   Log file            : {log_path}")
    print(f"\nNext step: run evaluate_attacks.py with MODEL_VARIANT='{MODEL_VARIANT}' "
          f"to compare clean vs robust performance.")

if __name__ == "__main__":
    main()
