import os
import hashlib
from pathlib import Path

SPLIT_ROOT = "/home/alucard-00/EC499/Project_Resourse/archive/malex_dataset"


def get_hash_to_paths(split, cls):
    folder = os.path.join(SPLIT_ROOT, split, cls)
    result = {}
    for fname in os.listdir(folder):
        if fname.endswith('.png'):
            fpath = os.path.join(folder, fname)
            with open(fpath, 'rb') as f:
                h = hashlib.md5(f.read()).hexdigest()
            result.setdefault(h, []).append(fpath)
    return result


print("Computing hashes for all splits...")
hashes = {}
for split in ['train', 'val', 'test']:
    for cls in ['benign', 'malware']:
        hashes[(split, cls)] = get_hash_to_paths(split, cls)
        unique = len(hashes[(split, cls)])
        print(f"  {split}/{cls}: {unique} unique hashes")

# Find and REMOVE files that appear in val or test but also in train
removed = 0
for cls in ['benign', 'malware']:
    train_h = hashes[('train', cls)]
    for split in ['val', 'test']:
        other_h = hashes[(split, cls)]
        overlap = set(train_h.keys()) & set(other_h.keys())
        print(f"\ntrain/{cls} ∩ {split}/{cls}: {len(overlap)} overlapping files")
        for h in overlap:
            for path_to_remove in other_h[h]:
                print(f"  REMOVING from {split}/{cls}: {os.path.basename(path_to_remove)}")
                os.remove(path_to_remove)
                removed += 1

print(f"\nTotal files removed: {removed}")
print("Re-verifying counts after cleanup...")
for split in ['train', 'val', 'test']:
    for cls in ['benign', 'malware']:
        folder = os.path.join(SPLIT_ROOT, split, cls)
        count = len([f for f in os.listdir(folder) if f.endswith('.png')])
        print(f"  {split}/{cls}: {count}")

print("\nTask A: Hash overlap remediation COMPLETE")
