import torch
import matplotlib.pyplot as plt
from dataset_loader import get_data_loaders
from config import MALIMG_ARCHIVE_DIR_STR
import os

def check_duplicates():
    print("🔬 RUNNING EXPERIMENT 3: VISUAL INSPECTION & DUPLICATE FILENAME CHECK")
    print("-" * 50)
    
    train_loader, val_loader, test_loader, _ = get_data_loaders(MALIMG_ARCHIVE_DIR_STR, batch_size=32)
    
    train_files = set(os.path.basename(f) for f, l in train_loader.dataset.samples)
    val_files = set(os.path.basename(f) for f, l in val_loader.dataset.samples)
    test_files = set(os.path.basename(f) for f, l in test_loader.dataset.samples)
    
    print(f"Train unique filenames (baseline): {len(train_files)}")
    print(f"Val unique filenames:            {len(val_files)}")
    
    overlap = train_files.intersection(val_files)
    print(f"OVERLAP FILENAMES (Train-Val):   {len(overlap)}")
    
    if len(overlap) > 0:
        print(f"❌ FILENAME OVERLAP DETECTED! Found {len(overlap)} duplicates.")
        print(f"Example: {list(overlap)[:5]}")
    else:
        print("✅ NO FILENAME OVERLAP BETWEEN SPLITS.")

    # Let's also compare the images themselves (first 10 of each)
    fig, axes = plt.subplots(4, 8, figsize=(20, 10))
    axes = axes.flatten()
    
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    
    tr_imgs, tr_labels = next(train_iter)
    va_imgs, va_labels = next(val_iter)
    
    for i in range(16):
        axes[i].imshow(tr_imgs[i].squeeze(), cmap='gray')
        axes[i].set_title(f"Train L:{tr_labels[i]}")
        axes[i].axis('off')
        
        axes[i+16].imshow(va_imgs[i].squeeze(), cmap='gray')
        axes[i+16].set_title(f"Val L:{va_labels[i]}")
        axes[i+16].axis('off')
        
    plt.tight_layout()
    plt.savefig('Project_Resourse/visual_inspection_batch.png')
    print("[*] Visual comparison saved to Project_Resourse/visual_inspection_batch.png")

if __name__ == "__main__":
    check_duplicates()
