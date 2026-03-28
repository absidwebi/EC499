import os
import torch
import numpy as np
import warnings
from PIL import Image
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, batch_size=32, num_workers=0):
    """
    Creates PyTorch DataLoaders for Train, Val, and Test splits.
    Automatically applies Binary Classification labeling:
        - 0: Benign
        - 1: Malware (All other classes)
    
    Args:
        data_dir (str): Path to the root dataset directory (e.g., malimg_dataset)
        batch_size (int): Number of images per batch
        num_workers (int): Number of subprocesses for data loading
        
    Returns:
        train_loader, val_loader, test_loader, class_weights_tensor
    """
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    # Reduce log spam from very large Malimg images.
    Image.MAX_IMAGE_PIXELS = None
    warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

    t_list = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),  # -> [-1,1]
    ]

    data_transforms = transforms.Compose(t_list)

    # 1. Load the raw datasets using ImageFolder
    # ImageFolder assigns an integer label based on alphabetical folder order
    raw_train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms)
    raw_val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms)
    raw_test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms)

    # 2. Setup Binary Labeling (Target Transform)
    # CRITICAL: Each split must compute its OWN benign index.
    # Val and test may have different alphabetical ordering than train
    # if any malware family folders are missing (e.g. Yuner.A was removed
    # from val/test after deduplication left it empty). Using the train
    # index for all splits causes benign images in val/test to be silently
    # mislabeled as malware.
    for split_name, split_ds in [('train', raw_train_dataset),
                                  ('val',   raw_val_dataset),
                                  ('test',  raw_test_dataset)]:
        if 'benign' not in split_ds.classes:
            raise ValueError(
                f"Error: 'benign' class folder not found in the {split_name} split. "
                f"Classes found: {split_ds.classes}"
            )
        idx = split_ds.classes.index('benign')
        split_ds.target_transform = lambda x, i=idx: 0 if x == i else 1
        print(f"[*] {split_name.upper():5} — benign at raw index {idx} "
              f"({len(split_ds.classes)} classes) -> binary transform applied")

    # 3. Create DataLoaders
    # Train loader is shuffled so the CNN sees a random mix of classes each batch
    train_loader = DataLoader(raw_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    # Val and Test loaders are not shuffled (order doesn't matter for evaluation, and keeps things deterministic)
    val_loader = DataLoader(raw_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(raw_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 4. Calculate Class Weights (To fix Dataset Imbalance)
    print("[*] Calculating Class Weights for CrossEntropyLoss...")
    
    # Iterate through the targets and apply our binary logic to get the true counts
    tt = raw_train_dataset.target_transform
    if tt is None:
        raise RuntimeError("Internal error: train target_transform not set")
    binary_targets = [tt(t) for t in raw_train_dataset.targets]
    
    benign_count = binary_targets.count(0)
    malware_count = binary_targets.count(1)
    total_samples = len(binary_targets)
    
    print(f"   - Training Samples: {total_samples}")
    print(f"   - Benign (Class 0): {benign_count}")
    print(f"   - Malware (Class 1): {malware_count}")

    # Standard formula for class weights: Total_Samples / (Number_of_Classes * Class_Count)
    weight_for_0 = total_samples / (2.0 * benign_count)
    weight_for_1 = total_samples / (2.0 * malware_count)
    
    # Create a PyTorch tensor containing the weights [weight_0, weight_1]
    # This must be moved to the GPU during training if your model is on the GPU.
    class_weights = torch.tensor([weight_for_0, weight_for_1], dtype=torch.float)
    
    print(f"   - Weight for Benign (0): {weight_for_0:.4f}")
    print(f"   - Weight for Malware (1): {weight_for_1:.4f}")

    return train_loader, val_loader, test_loader, class_weights

if __name__ == "__main__":
    from config import MALIMG_ARCHIVE_DIR_STR
    # Quick Test Execution
    print("Testing dataset_loader.py configuration...\n")
    DIR = MALIMG_ARCHIVE_DIR_STR
    
    try:
        train_l, val_l, test_l, weights = get_data_loaders(DIR, batch_size=16)
        
        print("\n✅ DataLoaders created successfully!")
        print(f"Number of training batches: {len(train_l)}")
        print(f"Number of validation batches: {len(val_l)}")
        print(f"Number of test batches: {len(test_l)}")
        
        # Test fetching one batch
        images, labels = next(iter(train_l))
        print(f"\nSample Batch Test:")
        print(f"Image tensor shape: {images.shape}  [BatchSize, Channels, Height, Width]")
        print(f"Label tensor shape: {labels.shape}")
        print(f"Sample labels (should only be 0s and 1s): {labels.tolist()}")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
