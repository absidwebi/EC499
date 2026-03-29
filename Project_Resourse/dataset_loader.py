import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_data_loaders(data_dir, batch_size=32, num_workers=0):
    """
    Creates PyTorch DataLoaders for Train, Val, and Test splits.

    Applies Binary Classification labeling:
        0 = benign
        1 = malware

    Compatible with MaleX dataset (images are already 256x256, no padding needed).

    Args:
        data_dir  : Path to the root dataset directory containing train/val/test subfolders.
        batch_size: Number of images per batch.
        num_workers: MUST remain 0 on Linux to prevent CUDA fork OOM errors.

    Returns:
        train_loader, val_loader, test_loader, class_weights_tensor
    """

    train_dir = os.path.join(data_dir, 'train')
    val_dir   = os.path.join(data_dir, 'val')
    test_dir  = os.path.join(data_dir, 'test')

    # Transform pipeline for MaleX byteplot images (already 256x256, no padding required).
    # Grayscale   — ensures 1-channel input for our grayscale-adapted ResNet-18.
    # ToTensor    — converts PIL image [0,255] to float tensor [0.0, 1.0].
    # Normalize   — maps [0,1] to [-1,1]; attack clamp range must be [-1,1] throughout.
    data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load datasets using ImageFolder.
    # ImageFolder assigns integer labels alphabetically:
    #   'benign'  -> index 0
    #   'malware' -> index 1
    # This is consistent across all three splits since both folders exist in every split.
    raw_train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms)
    raw_val_dataset   = datasets.ImageFolder(val_dir,   transform=data_transforms)
    raw_test_dataset  = datasets.ImageFolder(test_dir,  transform=data_transforms)

    # Validate and apply binary label mapping for each split independently.
    # This is a safety check; with MaleX the order will always be benign=0, malware=1.
    for split_name, split_ds in [('train', raw_train_dataset),
                                  ('val',   raw_val_dataset),
                                  ('test',  raw_test_dataset)]:
        if 'benign' not in split_ds.classes:
            raise ValueError(
                f"'benign' folder not found in {split_name} split. "
                f"Classes found: {split_ds.classes}"
            )
        if 'malware' not in split_ds.classes:
            raise ValueError(
                f"'malware' folder not found in {split_name} split. "
                f"Classes found: {split_ds.classes}"
            )
        idx_benign  = split_ds.classes.index('benign')
        idx_malware = split_ds.classes.index('malware')
        split_ds.target_transform = lambda x, i=idx_benign: 0 if x == i else 1
        print(f"[*] {split_name.upper():5} — benign={idx_benign}, malware={idx_malware} "
              f"-> binary transform applied ({len(split_ds)} images)")

    # DataLoaders — num_workers MUST be 0 to prevent CUDA fork OOM on Linux.
    train_loader = DataLoader(raw_train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(raw_val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(raw_test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    # Calculate Class Weights to handle any remaining imbalance.
    print("[*] Calculating class weights...")
    binary_targets  = [raw_train_dataset.target_transform(t)
                       for t in raw_train_dataset.targets]
    benign_count    = binary_targets.count(0)
    malware_count   = binary_targets.count(1)
    total_samples   = len(binary_targets)

    print(f"   - Training total    : {total_samples}")
    print(f"   - Benign  (label 0) : {benign_count}")
    print(f"   - Malware (label 1) : {malware_count}")

    weight_for_0 = total_samples / (2.0 * benign_count)
    weight_for_1 = total_samples / (2.0 * malware_count)
    class_weights = torch.tensor([weight_for_0, weight_for_1], dtype=torch.float)

    print(f"   - Weight benign  : {weight_for_0:.4f}")
    print(f"   - Weight malware : {weight_for_1:.4f}")

    return train_loader, val_loader, test_loader, class_weights


if __name__ == "__main__":
    from config import MALEX_DATASET_DIR_STR
    print("Testing dataset_loader.py with MaleX dataset...\n")

    try:
        train_l, val_l, test_l, weights = get_data_loaders(
            MALEX_DATASET_DIR_STR, batch_size=16)

        print(f"\nDataLoaders created successfully!")
        print(f"  Train batches : {len(train_l)}")
        print(f"  Val batches   : {len(val_l)}")
        print(f"  Test batches  : {len(test_l)}")

        images, labels = next(iter(train_l))
        print(f"\nBatch shape  : {images.shape}   [B, C, H, W]")
        print(f"Label shape  : {labels.shape}")
        print(f"Sample labels: {labels.tolist()}")
        print(f"Pixel min    : {images.min():.4f}  (expected >= -1.0)")
        print(f"Pixel max    : {images.max():.4f}  (expected <= +1.0)")

        unique_labels = set(labels.tolist())
        assert unique_labels.issubset({0, 1}), f"Labels contain unexpected values: {unique_labels}"
        assert images.shape[1] == 1, f"Expected 1 channel, got {images.shape[1]}"
        assert images.shape[2] == 256 and images.shape[3] == 256, \
            f"Expected 256x256, got {images.shape[2]}x{images.shape[3]}"
        assert images.min() >= -1.1 and images.max() <= 1.1, \
            f"Pixel values out of [-1,1] range"

        print("\nAll assertions PASSED")
        print("dataset_loader.py: READY FOR TRAINING")

    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()
