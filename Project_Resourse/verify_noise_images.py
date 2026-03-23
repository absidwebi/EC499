import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from dataset_loader import get_data_loaders
from models import get_resnet18_grayscale
from config import MALIMG_ARCHIVE_DIR_STR

class NoiseDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        
    def __len__(self):
        return len(self.original_dataset)
        
    def __getitem__(self, idx):
        # Return random noise instead of real image
        # labels are untouched
        _, label = self.original_dataset[idx]
        return torch.randn(1, 256, 256), label

def train_noise_images():
    print("🔬 RUNNING EXPERIMENT 2: RANDOM NOISE IMAGES")
    print("-" * 50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load loaders
    tr_l, va_l, _, _ = get_data_loaders(MALIMG_ARCHIVE_DIR_STR, batch_size=32)
    
    # Wrap with noise
    train_ds = NoiseDataset(tr_l.dataset)
    val_ds = NoiseDataset(va_l.dataset)
    
    print(f"[*] Initializing Dataloaders (Train: {len(train_ds)} samples, Val: {len(val_ds)} samples)...")
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
    
    # Get weights to counteract imbalance
    train_ds_raw = tr_l.dataset
    binary_targets = [train_ds_raw.target_transform(t) for t in train_ds_raw.targets] if hasattr(train_ds_raw, 'target_transform') else train_ds_raw.targets
    benign_count = binary_targets.count(0)
    malware_count = binary_targets.count(1)
    pos_weight = torch.tensor([benign_count / malware_count]).to(device)

    # Model
    model = get_resnet18_grayscale().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    print("[*] Training for 1 epoch with RANDOM NOISE images...")
    model.train()
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.float().to(device).unsqueeze(1)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        predicted = (outputs > 0).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if (batch_idx + 1) % 50 == 0:
            print(f"Batch {batch_idx+1}, Current Train Acc: {(correct/total)*100:.2f}%")
            
    print(f"\n[!] Final Noise Train Acc: {(correct/total)*100:.2f}%")
    
    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.float().to(device).unsqueeze(1)
            outputs = model(images)
            predicted = (outputs > 0).float()
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    print(f"[!] Final Noise Val Acc: {(val_correct/val_total)*100:.2f}%")
    print("\nCONCLUSION:")
    if val_correct/val_total > 0.95:
         print("❌ CRITICAL: The model learns even from noise! Labels are leaked in the metadata/order.")
    else:
         print("✅ PASS: Model cannot learn from noise (stays at baseline acc).")

if __name__ == "__main__":
    train_noise_images()
