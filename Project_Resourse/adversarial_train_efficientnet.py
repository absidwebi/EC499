"""
adversarial_train_efficientnet.py — Stage 3 Part 2 (EfficientNet-B0)
=====================================================================
Adversarial training of EfficientNet-B0 using identical PGD parameters
as adversarial_train.py for a controlled comparison with ResNet-18.

Output: efficientnet_b0_adversarially_trained.pth
        logs/adversarial_training_efficientnet.log
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    MALIMG_ARCHIVE_DIR_STR,
    EFFICIENTNET_CLEAN_MODEL_PATH_STR,
    EFFICIENTNET_ADV_TRAINED_MODEL_PATH_STR,
    LOGS_DIR,
)
from dataset_loader import get_data_loaders
from models import get_efficientnet_b0_grayscale

# === CONFIGURATION — identical to adversarial_train.py ===
BATCH_SIZE      = 16
NUM_EPOCHS      = 5
LEARNING_RATE   = 1e-4
ADV_TRAIN_EPS   = 0.05
ADV_TRAIN_ALPHA = 0.01
ADV_TRAIN_STEPS = 7
LOG_PATH        = LOGS_DIR / "adversarial_training_efficientnet.log"
# =========================================================

_criterion = nn.BCEWithLogitsLoss()


def pgd_attack(model, images, labels, eps, alpha, steps):
    """PGD attack — identical implementation to adversarial_train.py."""
    labels_f   = labels.float().unsqueeze(1)
    adv_images = images.clone().detach()
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
    adv_images = torch.clamp(adv_images, -1.0, 1.0).detach()
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


def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.float().to(device)
            predicted = (model(images) > 0).float().squeeze(1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)
    return 100.0 * correct / total


def evaluate_robust_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        adv = pgd_attack(model, images, labels,
                         eps=ADV_TRAIN_EPS, alpha=ADV_TRAIN_ALPHA, steps=20)
        with torch.no_grad():
            predicted = (model(adv) > 0).float().squeeze(1)
            correct += (predicted == labels.float()).sum().item()
            total   += labels.size(0)
    return 100.0 * correct / total


def main():
    print("=" * 60)
    print("  Stage 3 — EfficientNet-B0 Adversarial Training")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"[*] Device: {device}")
    print(f"[*] PGD: ε={ADV_TRAIN_EPS}, α={ADV_TRAIN_ALPHA}, steps={ADV_TRAIN_STEPS}")

    train_loader, val_loader, _, class_weights = get_data_loaders(
        data_dir=MALIMG_ARCHIVE_DIR_STR, batch_size=BATCH_SIZE, num_workers=0
    )
    class_weights = class_weights.to(device)

    print(f"[*] Loading clean EfficientNet-B0 weights: {EFFICIENTNET_CLEAN_MODEL_PATH_STR}")
    model = get_efficientnet_b0_grayscale().to(device)
    model.load_state_dict(torch.load(EFFICIENTNET_CLEAN_MODEL_PATH_STR, map_location=device))

    pos_weight = torch.tensor([class_weights[1] / class_weights[0]]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"[*] pos_weight: {pos_weight.item():.4f}")
    print("-" * 60)

    best_robust_val_acc = 0.0
    log_lines           = ["Epoch | Train Loss | Train Acc | Val Clean | Val Robust\n"]
    start_time          = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        model.train()
        train_loss = train_correct = train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            model.eval()
            adv_images = pgd_attack(model, images, labels,
                                    eps=ADV_TRAIN_EPS,
                                    alpha=ADV_TRAIN_ALPHA,
                                    steps=ADV_TRAIN_STEPS)
            model.train()
            optimizer.zero_grad()
            labels_f = labels.float().unsqueeze(1)
            outputs  = model(adv_images)
            loss     = criterion(outputs, labels_f)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * images.size(0)
            predicted      = (outputs > 0).float()
            train_total   += labels_f.size(0)
            train_correct += (predicted == labels_f).sum().item()

            if (batch_idx + 1) % 5 == 0:
                avg_t   = (time.time() - epoch_start) / (batch_idx + 1)
                est_rem = avg_t * (len(train_loader) - (batch_idx + 1))
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}  Est: {est_rem/60:.1f}m rem")

        epoch_train_loss = train_loss / train_total
        epoch_train_acc  = 100.0 * train_correct / train_total
        clean_val_acc    = evaluate_accuracy(model, val_loader, device)
        robust_val_acc   = evaluate_robust_accuracy(model, val_loader, device)

        log_line = (f"Epoch {epoch+1}/{NUM_EPOCHS} | "
                    f"Train Loss: {epoch_train_loss:.4f} | "
                    f"Train Acc: {epoch_train_acc:.2f}% | "
                    f"Val Clean: {clean_val_acc:.2f}% | "
                    f"Val Robust: {robust_val_acc:.2f}%")
        print(f"\n--- {log_line}")
        log_lines.append(log_line + "\n")

        if robust_val_acc > best_robust_val_acc:
            print(f"  ⭐ Robust val improved → {robust_val_acc:.2f}%. Saving...")
            torch.save(model.state_dict(), EFFICIENTNET_ADV_TRAINED_MODEL_PATH_STR)
            best_robust_val_acc = robust_val_acc
        print("-" * 60)

    total_time = time.time() - start_time
    print(f"\n🎉 EfficientNet-B0 AT complete in {total_time/60:.2f} minutes")
    print(f"   Best Robust Val Acc: {best_robust_val_acc:.2f}%")
    print(f"   Saved: {EFFICIENTNET_ADV_TRAINED_MODEL_PATH_STR}")

    with open(LOG_PATH, "w") as f:
        f.writelines(log_lines)
    print(f"   Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
