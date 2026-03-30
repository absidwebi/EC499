"""
Fine-tune ImageNet pretrained ResNet-18 on MaleX dataset.
Experiment B2: pretrained initialization to prevent early overfitting.
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
from models import get_resnet18_pretrained_grayscale
from config import (
    MALEX_DATASET_DIR_STR,
    RESNET_MALEX_PRETRAINED_CLEAN_PATH_STR,
    LOGS_DIR,
)

DATA_DIR = MALEX_DATASET_DIR_STR
MODEL_SAVE_PATH = RESNET_MALEX_PRETRAINED_CLEAN_PATH_STR
CHECKPOINT_DIR = os.path.dirname(MODEL_SAVE_PATH)
FULL_CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR, "resnet18_malex_pretrained_full_checkpoint.pth"
)

BATCH_SIZE = 32
NUM_EPOCHS = 50
# Lower LR for pretrained fine-tuning to avoid destroying learned features
LEARNING_RATE = 5e-5
EARLY_STOP_PATIENCE = 5
WEIGHT_DECAY = 1e-4
RESUME_IF_CHECKPOINT_EXISTS = True


def save_full_checkpoint(
    epoch,
    model,
    optimizer,
    scheduler,
    best_val_loss,
    best_epoch,
    epochs_no_improve,
    train_accs,
    val_accs,
    elapsed_seconds,
    path,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "epochs_no_improve": epochs_no_improve,
            "train_accs": train_accs,
            "val_accs": val_accs,
            "elapsed_seconds": elapsed_seconds,
        },
        path,
    )


def load_full_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])

    print(
        f"[*] Resumed from checkpoint: epoch {ckpt['epoch']+1}, "
        f"best_val_loss={ckpt['best_val_loss']:.4f} (epoch {ckpt['best_epoch']})"
    )

    return {
        "start_epoch": ckpt["epoch"] + 1,
        "best_val_loss": ckpt["best_val_loss"],
        "best_epoch": ckpt["best_epoch"],
        "epochs_no_improve": ckpt.get("epochs_no_improve", 0),
        "train_accs": ckpt.get("train_accs", []),
        "val_accs": ckpt.get("val_accs", []),
        "elapsed_seconds": ckpt.get("elapsed_seconds", 0.0),
    }


def plot_curves(train_accs, val_accs, save_path):
    epochs = range(1, len(train_accs) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accs, 'b-o', label='Train Accuracy', linewidth=2)
    plt.plot(epochs, val_accs, 'r-o', label='Val Accuracy', linewidth=2)
    plt.title('ResNet-18 Pretrained Fine-tune - MaleX', fontsize=14)
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
    print("  Experiment B2: ResNet-18 Pretrained Fine-tuning on MaleX")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    print("[*] Loading MaleX dataset...")
    train_loader, val_loader, _, class_weights = get_data_loaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )
    class_weights = class_weights.to(device)

    print("[*] Loading ResNet-18 with ImageNet pretrained weights...")
    model = get_resnet18_pretrained_grayscale().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Total parameters: {total_params:,}")

    pos_weight = torch.tensor([class_weights[1] / class_weights[0]]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    print(f"[*] Optimizer   : AdamW (LR={LEARNING_RATE}, WD={WEIGHT_DECAY})")
    print(f"[*] Loss        : BCEWithLogitsLoss (pos_weight={pos_weight.item():.4f})")
    print(f"[*] Batch size  : {BATCH_SIZE}")
    print(f"[*] Early stop  : patience={EARLY_STOP_PATIENCE}")
    print("-" * 60)

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    train_accs, val_accs = [], []
    start_epoch = 0
    previous_elapsed_seconds = 0.0

    if RESUME_IF_CHECKPOINT_EXISTS and os.path.exists(FULL_CHECKPOINT_PATH):
        resume_data = load_full_checkpoint(
            FULL_CHECKPOINT_PATH, model, optimizer, scheduler, device
        )
        start_epoch = resume_data["start_epoch"]
        best_val_loss = resume_data["best_val_loss"]
        best_epoch = resume_data["best_epoch"]
        epochs_no_improve = resume_data["epochs_no_improve"]
        train_accs = resume_data["train_accs"]
        val_accs = resume_data["val_accs"]
        previous_elapsed_seconds = resume_data["elapsed_seconds"]
    else:
        print("[*] Starting fresh training run.")

    curve_path = os.path.join(str(LOGS_DIR), "training_curve_resnet_pretrained.png")
    start_time = time.time()

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            predicted = (outputs > 0).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            if (batch_idx + 1) % 50 == 0:
                running_acc = 100.0 * train_correct / train_total
                print(
                    f"  Epoch [{epoch+1}/{NUM_EPOCHS}] "
                    f"Step [{batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}  Running Acc: {running_acc:.2f}%"
                )

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = 100.0 * train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                predicted = (outputs > 0).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100.0 * val_correct / val_total
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)

        elapsed = (previous_elapsed_seconds + (time.time() - start_time)) / 60
        print(f"\n{'='*60}")
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS}  ({elapsed:.1f} min elapsed)")
        print(f"  Train  - Loss: {epoch_train_loss:.4f}  Acc: {epoch_train_acc:.2f}%")
        print(f"  Val    - Loss: {epoch_val_loss:.4f}  Acc: {epoch_val_acc:.2f}%")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  LR now: {current_lr:.2e}")

        if epoch_val_loss < best_val_loss:
            improvement = best_val_loss - epoch_val_loss
            print(f"  STAR Val loss improved by {improvement:.4f} -> saving model")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            best_val_loss = epoch_val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{EARLY_STOP_PATIENCE} epochs")

        scheduler.step(epoch_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            print(f"  LR reduced: {current_lr:.2e} -> {new_lr:.2e}")

        current_elapsed_seconds = previous_elapsed_seconds + (time.time() - start_time)
        save_full_checkpoint(
            epoch,
            model,
            optimizer,
            scheduler,
            best_val_loss,
            best_epoch,
            epochs_no_improve,
            train_accs,
            val_accs,
            current_elapsed_seconds,
            FULL_CHECKPOINT_PATH,
        )
        print(f"  Full checkpoint saved (epoch {epoch+1})")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}. Best was epoch {best_epoch}.")
            break
        print(f"{'='*60}\n")

    total_time = time.time() - start_time
    print(f"\nPretrained ResNet-18 training complete in {total_time/60:.2f} minutes")
    print(f"   Best val loss : {best_val_loss:.4f}  (epoch {best_epoch})")
    print(f"   Weights saved : {MODEL_SAVE_PATH}")
    print(f"   Full ckpt     : {FULL_CHECKPOINT_PATH}")
    plot_curves(train_accs, val_accs, curve_path)


if __name__ == "__main__":
    train()
