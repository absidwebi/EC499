import os
import shutil
import random
import math
from pathlib import Path

BENIGN_SRC  = "/home/alucard-00/EC499/ben_byteplot_imgs_zipped/byteplot_imgs_RxR/256"
MALWARE_SRC = "/home/alucard-00/EC499/mal_byteplot_imgs_zipped/byteplot_imgs_RxR/256"
OUTPUT_ROOT = "/home/alucard-00/EC499/Project_Resourse/archive/malex_dataset"

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10
RANDOM_SEED = 42

def get_png_paths(folder):
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith('.png')
    ])

def stratified_split(files, train_r, val_r, seed):
    random.seed(seed)
    files_shuffled = files.copy()
    random.shuffle(files_shuffled)
    n = len(files_shuffled)
    n_train = math.floor(n * train_r)
    n_val   = math.floor(n * val_r)
    n_test  = n - n_train - n_val
    train = files_shuffled[:n_train]
    val   = files_shuffled[n_train:n_train + n_val]
    test  = files_shuffled[n_train + n_val:]
    return train, val, test

def copy_files(file_list, dest_dir, class_name, split_name):
    os.makedirs(dest_dir, exist_ok=True)
    for i, src in enumerate(file_list):
        fname = os.path.basename(src)
        dst = os.path.join(dest_dir, fname)
        # Handle potential filename collisions with a counter prefix
        if os.path.exists(dst):
            base, ext = os.path.splitext(fname)
            dst = os.path.join(dest_dir, f"{base}_{i}{ext}")
        shutil.copy2(src, dst)
    print(f"  [{split_name}/{class_name}] Copied {len(file_list)} files")

print("=" * 60)
print("MaleX Balanced Split Builder")
print("=" * 60)

# 1. Load file paths
benign_files  = get_png_paths(BENIGN_SRC)
malware_files = get_png_paths(MALWARE_SRC)
print(f"Benign files found:  {len(benign_files)}")
print(f"Malware files found: {len(malware_files)}")

# 2. Downsample malware to match benign count
random.seed(RANDOM_SEED)
if len(malware_files) > len(benign_files):
    malware_files_balanced = random.sample(malware_files, len(benign_files))
    print(f"Malware downsampled to: {len(malware_files_balanced)} (matched to benign count)")
else:
    malware_files_balanced = malware_files
    print(f"No downsampling needed. Malware count: {len(malware_files_balanced)}")

# 3. Stratified split per class
b_train, b_val, b_test = stratified_split(benign_files,           TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)
m_train, m_val, m_test = stratified_split(malware_files_balanced, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)

print(f"\nSplit sizes:")
print(f"  Train: {len(b_train)} benign + {len(m_train)} malware = {len(b_train)+len(m_train)}")
print(f"  Val:   {len(b_val)}   benign + {len(m_val)}   malware = {len(b_val)+len(m_val)}")
print(f"  Test:  {len(b_test)}  benign + {len(m_test)}  malware = {len(b_test)+len(m_test)}")
total = len(b_train)+len(m_train)+len(b_val)+len(m_val)+len(b_test)+len(m_test)
print(f"  TOTAL: {total}")

# 4. Copy files
print(f"\nCopying files to {OUTPUT_ROOT} ...")
copy_files(b_train, os.path.join(OUTPUT_ROOT, "train", "benign"),  "benign",  "train")
copy_files(m_train, os.path.join(OUTPUT_ROOT, "train", "malware"), "malware", "train")
copy_files(b_val,   os.path.join(OUTPUT_ROOT, "val",   "benign"),  "benign",  "val")
copy_files(m_val,   os.path.join(OUTPUT_ROOT, "val",   "malware"), "malware", "val")
copy_files(b_test,  os.path.join(OUTPUT_ROOT, "test",  "benign"),  "benign",  "test")
copy_files(m_test,  os.path.join(OUTPUT_ROOT, "test",  "malware"), "malware", "test")

# 5. Verify counts match
print("\nVerifying output folder counts...")
for split in ["train", "val", "test"]:
    for cls in ["benign", "malware"]:
        dest = os.path.join(OUTPUT_ROOT, split, cls)
        count = len([f for f in os.listdir(dest) if f.endswith('.png')])
        print(f"  {split}/{cls}: {count}")

print("\nPHASE 2: COMPLETE")
