import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import random
import numpy as np
from dataset_loader import get_data_loaders
from models import get_resnet18_grayscale
from config import MALIMG_ARCHIVE_DIR_STR

def train_shuffled_labels():
    print("🔬 RUNNING EXPERIMENT 1: LABEL SHUFFLING")
    print("-" * 50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load loaders
    train_loader, val_loader, test_loader, class_weights = get_data_loaders(MALIMG_ARCHIVE_DIR_STR, batch_size=32)
    
    # Apply label shuffling to the *underlying* dataset
    # Note: raw_train_dataset is accessed via dataset
    train_ds = train_loader.dataset
    val_ds = val_loader.dataset
    
    # ImageFolder uses self.samples = [(path, label), ...]
    print("[*] Shuffling labels in self.samples...")
    
    def shuffle_ds(ds):
        samples = list(ds.samples)
        # Separate labels and shuffle them
        paths = [s[0] for s in samples]
        labels = [s[1] for s in samples]
        random.shuffle(labels)
        # Reconstruct samples
        ds.samples = list(zip(paths, labels))
        # Keep internal labels list in sync (used by some libraries)
        ds.targets = labels
        
    shuffle_ds(train_ds)
    shuffle_ds(val_ds)
    
    # Get weights to counteract imbalance
    binary_targets = [train_ds.target_transform(t) for t in train_ds.targets] if hasattr(train_ds, 'target_transform') else train_ds.targets
    benign_count = binary_targets.count(0)
    malware_count = binary_targets.count(1)
    pos_weight = torch.tensor([benign_count / malware_count]).to(device)

    # Model
    model = get_resnet18_grayscale().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Train for ONE epoch
    print("[*] Training for 1 epoch with GARBAGE labels...")
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
            
    print(f"\n[!] Final Shuffled Train Acc: {(correct/total)*100:.2f}%")
    print("-" * 50)
    
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
            
    print(f"[!] Final Shuffled Val Acc: {(val_correct/val_total)*100:.2f}%")
    print("\nCONCLUSION:")
    if val_correct/val_total > 0.60:
        print("❌ CRITICAL: The model is learning even with random labels! Leakage or serious bug confirmed.")
    else:
        print("✅ PASS: Model cannot learn from garbage labels (50-60% acc). Pipeline integrity confirmed.")

if __name__ == "__main__":
    train_shuffled_labels()
