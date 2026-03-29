import os
import hashlib
from pathlib import Path

DATASET_ROOT = "/home/alucard-00/EC499/Project_Resourse/archive/malex_dataset"


def get_hashes(split, cls):
    folder = os.path.join(DATASET_ROOT, split, cls)
    hashes = set()
    for f in os.listdir(folder):
        if f.endswith('.png'):
            path = os.path.join(folder, f)
            with open(path, 'rb') as fh:
                hashes.add(hashlib.md5(fh.read()).hexdigest())
    return hashes


print("Computing hashes for all splits and classes...")
splits = ["train", "val", "test"]
classes = ["benign", "malware"]

all_hashes = {}
for s in splits:
    for c in classes:
        all_hashes[(s, c)] = get_hashes(s, c)
        print(f"  {s}/{c}: {len(all_hashes[(s,c)])} unique hashes")

print("\nChecking for cross-split overlaps...")
failed = False
for c in classes:
    tr = all_hashes[("train", c)]
    va = all_hashes[("val", c)]
    te = all_hashes[("test", c)]
    tv = tr & va
    tt = tr & te
    vt = va & te
    print(f"  {c}: train&val={len(tv)}, train&test={len(tt)}, val&test={len(vt)}")
    if len(tv) + len(tt) + len(vt) > 0:
        failed = True
        print(f"  FAIL: Overlap detected in {c}")

if not failed:
    print("\nCheck 6A: PASS - Zero cross-split hash overlaps")
else:
    print("\nCheck 6A: FAIL - Data leakage detected - DO NOT PROCEED")
