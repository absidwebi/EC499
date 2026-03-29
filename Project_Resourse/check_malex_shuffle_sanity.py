import torch
import torch.nn as nn
import torch.optim as optim
import random
from config import MALEX_DATASET_DIR_STR
from dataset_loader import get_data_loaders
from models import get_resnet18_grayscale

print("Label Shuffle Sanity Test (3 mini-epochs on shuffled labels)...")
print("Expected result: accuracy near 50% +/- 10%")
print("If accuracy >> 60%, a geometric shortcut still exists.\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

train_l, val_l, _, _ = get_data_loaders(MALEX_DATASET_DIR_STR, batch_size=64)

model = get_resnet18_grayscale().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(3):
    model.train()
    correct = 0
    total = 0
    for images, labels in train_l:
        # Randomly shuffle labels - destroy all real signal
        shuffled = labels[torch.randperm(labels.size(0))]
        images = images.to(device)
        shuffled = shuffled.float().unsqueeze(1).to(device)

        outputs = model(images)
        loss = criterion(outputs, shuffled)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted = (outputs > 0).float()
        correct += (predicted == shuffled).sum().item()
        total += shuffled.size(0)

    train_acc = 100.0 * correct / total

    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_l:
            shuffled = labels[torch.randperm(labels.size(0))]
            images = images.to(device)
            shuffled = shuffled.float().unsqueeze(1).to(device)
            outputs = model(images)
            predicted = (outputs > 0).float()
            val_correct += (predicted == shuffled).sum().item()
            val_total += shuffled.size(0)

    val_acc = 100.0 * val_correct / val_total
    print(f"  Epoch {epoch+1}: Train Acc = {train_acc:.1f}%  |  Val Acc = {val_acc:.1f}%")

final_val_acc = val_acc
if final_val_acc <= 60.0:
    print(f"\nCheck 6C: PASS - Shuffle val accuracy {final_val_acc:.1f}% is near 50%")
    print("No shortcut detected. Dataset is clean.")
else:
    print(f"\nCheck 6C: WARN - Shuffle val accuracy {final_val_acc:.1f}% is high")
    print("Investigate before proceeding to training.")
