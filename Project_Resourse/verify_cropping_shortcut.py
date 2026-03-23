import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import get_resnet18_grayscale
from config import MALIMG_ARCHIVE_DIR_STR
import os

def train_unbiased():
    print("🔬 EXPERIMENT 4: UNBIASED PADDING/CROP VS RESIZE")
    print("-" * 50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Unbiased Transform: No Resizing.
    # We use CenterCrop(256). If images are smaller than 256, we must pad.
    # A standard way to pad to a fixed size without resizing is:
    unbiased_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop(256), # This will crop 256x256 from the center of the original image
        # Note: If image is smaller than 256, CenterCrop pads with 0s by default.
        transforms.ToTensor(),
    ])
    
    base = MALIMG_ARCHIVE_DIR_STR
    train_ds = datasets.ImageFolder(os.path.join(base, 'train'), transform=unbiased_transform)
    val_ds = datasets.ImageFolder(os.path.join(base, 'val'), transform=unbiased_transform)
    
    # Binary Labeling logic (Benign=0, Else=1)
    benign_idx = train_ds.classes.index('benign')
    train_ds.target_transform = lambda x: 0 if x == benign_idx else 1
    val_ds.target_transform = lambda x: 0 if x == benign_idx else 1
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    model = get_resnet18_grayscale().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    print("[*] Training for 1 epoch with 1:1 CROP (No Resizing artifacts)...")
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.float().to(device).unsqueeze(1)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (batch_idx + 1) % 50 == 0:
            predicted = (outputs > 0).float()
            acc = (predicted == labels).sum().item() / labels.size(0) * 100
            print(f"Batch {batch_idx+1}, Acc: {acc:.2f}%")
            
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
            
    print(f"\n[!] Final Unbiased Val Acc: {(val_correct/val_total)*100:.2f}%")
    print("-" * 50)
    print("ANALYSIS:")
    if val_correct/val_total < 0.90:
        print("🔍 ACCURACY DROPPED! This proves that the 99% accuracy was partially built on resizing artifacts (Structural Bias).")
    else:
        print("🌟 ACCURACY STAYS HIGH! This suggests the textures themselves are highly discriminative even without resizing artifacts.")

if __name__ == "__main__":
    train_unbiased()
