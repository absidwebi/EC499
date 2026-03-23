import torch
import matplotlib.pyplot as plt
from dataset_loader import get_data_loaders
from config import MALIMG_ARCHIVE_DIR_STR, LOGS_DIR
import os

def visualize_clean_batch():
    train_loader, _, _, _ = get_data_loaders(MALIMG_ARCHIVE_DIR_STR, batch_size=8)
    images, labels = next(iter(train_loader))
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for i in range(8):
        img = images[i].squeeze().numpy()
        label = labels[i].item()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Label: {'Malware' if label == 1 else 'Benign'}")
        axes[i].axis('off')
    
    plot_path = os.path.join(LOGS_DIR, "clean_dataset_check.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"[*] Visualization saved to {plot_path}")

if __name__ == "__main__":
    visualize_clean_batch()
