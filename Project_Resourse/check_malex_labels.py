import torch
from config import MALEX_DATASET_DIR_STR
from dataset_loader import get_data_loaders

print("Loading MaleX DataLoaders for label verification...")
train_l, val_l, test_l, weights = get_data_loaders(MALEX_DATASET_DIR_STR, batch_size=64)

for loader_name, loader in [("TRAIN", train_l), ("VAL", val_l), ("TEST", test_l)]:
    images, labels = next(iter(loader))
    benign_count = (labels == 0).sum().item()
    malware_count = (labels == 1).sum().item()
    unexpected = ((labels != 0) & (labels != 1)).sum().item()

    print(f"\n{loader_name} batch:")
    print(f"  Image shape  : {images.shape}")
    print(f"  Channels     : {images.shape[1]}  (expected: 1)")
    print(f"  Spatial      : {images.shape[2]}x{images.shape[3]}  (expected: 256x256)")
    print(f"  Pixel min    : {images.min():.4f}  (expected >= -1.0)")
    print(f"  Pixel max    : {images.max():.4f}  (expected <= +1.0)")
    print(f"  Benign  (0)  : {benign_count}")
    print(f"  Malware (1)  : {malware_count}")
    print(f"  Unexpected   : {unexpected}  (expected: 0)")

    assert images.shape[1] == 1, "FAIL: Wrong channel count"
    assert images.shape[2] == 256, "FAIL: Wrong height"
    assert images.shape[3] == 256, "FAIL: Wrong width"
    assert images.min() >= -1.1, "FAIL: Pixel min out of range"
    assert images.max() <= 1.1, "FAIL: Pixel max out of range"
    assert unexpected == 0, "FAIL: Unexpected label values"
    print(f"  {loader_name}: ALL ASSERTIONS PASS")

print("\nClass weights:", weights)
print("\nCheck 6B: PASS - Labels and tensors are correct")
