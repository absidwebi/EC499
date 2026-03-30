import os
import hashlib

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


removed = 0
for cls in ['benign', 'malware']:
    val_h = get_hash_to_paths('val', cls)
    test_h = get_hash_to_paths('test', cls)
    overlap = set(val_h.keys()) & set(test_h.keys())
    print(f"val/{cls} ∩ test/{cls}: {len(overlap)}")
    # Keep validation copy, remove from test to preserve train/val model selection set
    for h in overlap:
        for path_to_remove in test_h[h]:
            os.remove(path_to_remove)
            removed += 1
            print(f"  REMOVED from test/{cls}: {os.path.basename(path_to_remove)}")

print(f"Total val-test files removed: {removed}")
for split in ['train', 'val', 'test']:
    for cls in ['benign', 'malware']:
        folder = os.path.join(SPLIT_ROOT, split, cls)
        count = len([f for f in os.listdir(folder) if f.endswith('.png')])
        print(f"  {split}/{cls}: {count}")
