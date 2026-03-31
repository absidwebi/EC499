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

# === CONFIGURATION ===
BATCH_SIZE    = 16
NUM_EPOCHS    = 5 # 5 is enough for fine-tuning robustness
LEARNING_RATE = 1e-4

# PGD attack parameters used DURING adversarial training
# These must be strong enough to properly harden the model.
ADV_TRAIN_EPS   = 0.05   # Max perturbation budget
ADV_TRAIN_ALPHA = 0.01   # Step size per PGD iteration
ADV_TRAIN_STEPS = 7      # 7-step PGD (Madry Default)

# Model options: "3c2d", "resnet", "resnet_pretrained"
MODEL_VARIANT = "3c2d"

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
    print("  Stage 3 — Part 2: Adversarial Training Defense")
    print("=" * 60)

    # 1. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"[*] Using device: {device}")
    print(f"[*] Selected model variant: {MODEL_VARIANT}")
    print(f"[*] PGD config for training: ε={ADV_TRAIN_EPS}, α={ADV_TRAIN_ALPHA}, steps={ADV_TRAIN_STEPS}")

    # 2. Load Data
    print("[*] Loading dataset...")
    train_loader, val_loader, _, class_weights = get_data_loaders(
        data_dir=MALEX_DATASET_DIR_STR,
        batch_size=BATCH_SIZE,
        num_workers=0
    )
    class_weights = class_weights.to(device)

    # 3. Load Vulnerable Baseline and Start from Its Weights
    # (Warm-starting from a converged model saves time vs. training from scratch)
    model_name, model, clean_model_path, robust_model_save_path = get_model_bundle(MODEL_VARIANT)
    print(f"[*] Warm-starting {model_name} from clean model: {clean_model_path}")
    model = model.to(device)
    model.load_state_dict(torch.load(clean_model_path, map_location=device))

    # 4. Loss, Optimizer
    pos_weight = torch.tensor([class_weights[1] / class_weights[0]]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"[*] Optimizer: AdamW (LR: {LEARNING_RATE})")
    print(f"[*] Loss: BCEWithLogitsLoss (pos_weight: {pos_weight.item():.4f})")
    print("-" * 60)

    best_robust_val_acc = 0.0
    log_lines = ["Epoch | Train Loss | Train Acc | Val Clean Acc | Val Robust Acc\n"]
    start_time = time.time()
    print("[*] Entering training loop...")

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        # --- ADVERSARIAL TRAINING PHASE ---
        model.train()
        train_loss    = 0.0
        train_correct = 0
        train_total   = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Step A: Generate adversarial examples using current model weights
            model.eval()  # set eval so BN uses running stats during attack
            adv_images = pgd_attack(model, images, labels,
                                    eps=ADV_TRAIN_EPS,
                                    alpha=ADV_TRAIN_ALPHA,
                                    steps=ADV_TRAIN_STEPS)

            # Step B: Train on ADVERSARIAL examples (not the original clean images)
            model.train()
            optimizer.zero_grad()

            labels_f = labels.float().unsqueeze(1)
            outputs  = model(adv_images)
            loss     = criterion(outputs, labels_f)
            loss.backward()
            optimizer.step()

            # Metrics
            train_loss    += loss.item() * images.size(0)
            predicted      = (outputs > 0).float()
            train_total   += labels_f.size(0)
            train_correct += (predicted == labels_f).sum().item()

            if (batch_idx + 1) % 5 == 0:
                avg_time = (time.time() - epoch_start) / (batch_idx + 1)
                est_rem = avg_time * (len(train_loader) - (batch_idx + 1))
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Est: {est_rem/60:.1f}m rem")

        epoch_train_loss = train_loss / train_total
        epoch_train_acc  = (train_correct / train_total) * 100.0

        # --- VALIDATION PHASE ---
        clean_val_acc  = evaluate_accuracy(model, val_loader, device)
        robust_val_acc = evaluate_robust_accuracy_manual(model, val_loader, device)

        log_line = (
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Train Acc: {epoch_train_acc:.2f}% | "
            f"Val Clean: {clean_val_acc:.2f}% | "
            f"Val Robust: {robust_val_acc:.2f}%"
        )
        print(f"\n--- {log_line}")
        log_lines.append(log_line + "\n")

        # Save best model based on robust validation accuracy
        if robust_val_acc > best_robust_val_acc:
            print(f"  ⭐ Robust val acc improved ({best_robust_val_acc:.2f}% → {robust_val_acc:.2f}%). Saving model...")
            torch.save(model.state_dict(), robust_model_save_path)
            best_robust_val_acc = robust_val_acc

        print("-" * 60)

    total_time = time.time() - start_time
    print(f"\n🎉 Adversarial Training Complete in {total_time/60:.2f} minutes!")
    print(f"   Best Robust Val Accuracy: {best_robust_val_acc:.2f}%")
    print(f"   Robust Model saved to: {robust_model_save_path}")
    print("\n→ Run evaluate_attacks.py again on the new robust model to compare results.")

    # Write log
    model_tag = MODEL_VARIANT.lower().replace(" ", "_")
    log_path = LOGS_DIR / f"adversarial_training_log_{model_tag}.txt"
    with open(log_path, "w") as f:
        f.writelines(log_lines)
    print(f"   Training log saved to: {log_path}")

if __name__ == "__main__":
    main()
