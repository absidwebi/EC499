import os
import hashlib
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm

def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def fix_leakage(data_dir):
    data_dir = Path(data_dir)
    print(f"[*] Starting Dataset Leakage Fix for: {data_dir}")
    
    # 1. Collect all files from all splits
    all_files_by_family = defaultdict(list)
    for split in ["train", "val", "test"]:
        split_path = data_dir / split
        if not split_path.exists():
            continue
        for family_path in split_path.iterdir():
            if family_path.is_dir():
                family_name = family_path.name
                for img_path in family_path.glob("*.png"):
                    all_files_by_family[family_name].append(img_path)

    print(f"[*] Found {len(all_files_by_family)} families.")

    # 2. De-duplicate by hash and group by family
    unique_files_by_family = defaultdict(list)
    total_found = 0
    total_unique = 0
    global_hashes = set()

    for family, files in all_files_by_family.items():
        family_hashes = set()
        for f in files:
            h = file_hash(f)
            total_found += 1
            if h not in global_hashes:
                unique_files_by_family[family].append(f)
                global_hashes.add(h)
                total_unique += 1
    
    print(f"[*] Total files found: {total_found}")
    print(f"[*] Total unique files: {total_unique} (Removed {total_found - total_unique} duplicates)")

    # 3. Create fresh split directories
    new_data_dir = data_dir.parent / "malimg_dataset_clean"
    if new_data_dir.exists():
        shutil.rmtree(new_data_dir)
    
    for split in ["train", "val", "test"]:
        for family in unique_files_by_family.keys():
            (new_data_dir / split / family).mkdir(parents=True, exist_ok=True)

    # 4. Perform Stratified Split
    print("[*] Re-splitting data (80/10/10)...")
    for family, files in unique_files_by_family.items():
        if len(files) < 3:
            # Not enough to split, just put in train
            train_f, val_f, test_f = files, [], []
        else:
            train_f, temp_f = train_test_split(files, test_size=0.20, random_state=42)
            if len(temp_f) >= 2:
                val_f, test_f = train_test_split(temp_f, test_size=0.50, random_state=42)
            else:
                val_f, test_f = temp_f, []

        # Copy files
        for split, split_files in [("train", train_f), ("val", val_f), ("test", test_f)]:
            for f in split_files:
                dest = new_data_dir / split / family / f.name
                shutil.copy2(f, dest)

    print(f"[*] Clean dataset created at: {new_data_dir}")
    print("[*] Swapping directories...")
    
    # Backup old and move new
    old_backup = data_dir.parent / "malimg_dataset_leaked_backup"
    if old_backup.exists():
        shutil.rmtree(old_backup)
    
    shutil.move(data_dir, old_backup)
    shutil.move(new_data_dir, data_dir)
    
    print("✅ Dataset Fix Complete! No more overlaps.")

if __name__ == "__main__":
    from config import MALIMG_ARCHIVE_DIR_STR
    fix_leakage(MALIMG_ARCHIVE_DIR_STR)
