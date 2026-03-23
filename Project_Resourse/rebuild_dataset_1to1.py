import os
import shutil
import hashlib
import random
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- CONFIG ---
ROOT = Path("/home/alucard-00/EC499")
BENIGN_PE_DIR = ROOT / "benign_pe_files"
MALWARE_DATA_DIR = ROOT / "Project_Resourse" / "archive" / "malimg_dataset"
CLEAN_DATA_ROOT = ROOT / "Project_Resourse" / "archive" # Directory to place the clean dataset
TARGET_SIZE = (256, 256)

def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def pad_to_256(img_data):
    """Simple byte level padding for PE files."""
    target = 256 * 256
    if len(img_data) > target:
        return img_data[:target]
    return img_data + b"\x00" * (target - len(img_data))

def convert_pe_to_image(pe_path, output_path):
    with open(pe_path, "rb") as f:
        data = f.read(256 * 256) # Read first 65536 bytes
    
    padded_data = pad_to_256(data)
    from PIL import Image
    import numpy as np
    
    img_array = np.frombuffer(padded_data, dtype=np.uint8).reshape(256, 256)
    img = Image.fromarray(img_array, mode='L')
    img.save(output_path)

def main():
    print("🚀 Rebuilding 1:1 Balanced Dataset (32KB-512KB Filtering)")
    print("-" * 50)

    # 1. Collect Valid Benign PEs (32KB-512KB)
    print("[*] Filtering Benign PEs...")
    valid_benign = []
    for f in BENIGN_PE_DIR.iterdir():
        if f.is_file():
            size = f.stat().st_size
            if 32 * 1024 < size < 512 * 1024:
                valid_benign.append(f)
    
    total_benign = len(valid_benign)
    print(f"[*] Found {total_benign} valid benign files.")

    # 2. Collect Malware Samples from cleaned dataset
    print("[*] Collecting Malware proportionally...")
    malware_files_by_family = defaultdict(list)
    for family_path in (MALWARE_DATA_DIR / "train").iterdir():
        if family_path.is_dir() and family_path.name != "benign":
            malware_files_by_family[family_path.name].extend(list((MALWARE_DATA_DIR / "train" / family_path.name).glob("*.png")))
    for family_path in (MALWARE_DATA_DIR / "val").iterdir():
        if family_path.is_dir() and family_path.name != "benign":
            malware_files_by_family[family_path.name].extend(list((MALWARE_DATA_DIR / "val" / family_path.name).glob("*.png")))
    for family_path in (MALWARE_DATA_DIR / "test").iterdir():
        if family_path.is_dir() and family_path.name != "benign":
            malware_files_by_family[family_path.name].extend(list((MALWARE_DATA_DIR / "test" / family_path.name).glob("*.png")))

    total_malware_avail = sum(len(v) for v in malware_files_by_family.values())
    print(f"[*] Total malware available: {total_malware_avail}")

    # Aim for 1:1. Use min(total_benign, total_malware_avail)
    target_count = min(total_benign, total_malware_avail)
    print(f"[*] Target count for each class: {target_count}")

    # Randomly select Benign
    selected_benign = random.sample(valid_benign, target_count)

    # Proportional selection of Malware Families
    selected_malware = []
    for family, files in malware_files_by_family.items():
        # Count for this family: target_count * (family_count / total_malware_avail)
        family_target = int(target_count * (len(files) / total_malware_avail))
        family_target = min(family_target, len(files))
        selected_malware.append((family, random.sample(files, family_target)))

    # Fix rounding errors (if we have fewer than target_count)
    current_mal_count = sum(len(f[1]) for f in selected_malware)
    if current_mal_count < target_count:
        missing = target_count - current_mal_count
        # Add to the largest family
        largest_fam = max(malware_files_by_family.items(), key=lambda x: len(x[1]))
        remaining = [f for f in largest_fam[1] if f not in dict(selected_malware).get(largest_fam[0], [])]
        if len(remaining) >= missing:
            # Append to the list in selected_malware
            for i, (fam, files) in enumerate(selected_malware):
                if fam == largest_fam[0]:
                    selected_malware[i] = (fam, files + random.sample(remaining, missing))
                    break

    # 3. Prepare Directory
    new_dataset_dir = CLEAN_DATA_ROOT / "malimg_dataset_balanced"
    if new_dataset_dir.exists():
        shutil.rmtree(new_dataset_dir)
    
    for split in ["train", "val", "test"]:
        (new_dataset_dir / split / "benign").mkdir(parents=True, exist_ok=True)
        for family, _ in selected_malware:
            (new_dataset_dir / split / family).mkdir(parents=True, exist_ok=True)

    # 4. Process and Move Files
    print("[*] Processing Benign images...")
    # Split benign files
    b_tr, b_temp = train_test_split(selected_benign, test_size=0.20, random_state=42)
    b_val, b_test = train_test_split(b_temp, test_size=0.50, random_state=42)

    for split, files in [("train", b_tr), ("val", b_val), ("test", b_test)]:
        for f in tqdm(files, desc=f"Benign {split}"):
            dest = new_dataset_dir / split / "benign" / (f.stem + ".png")
            convert_pe_to_image(f, dest)

    print("[*] Copying Malware images...")
    for family, files in selected_malware:
        m_tr, m_temp = train_test_split(files, test_size=0.20, random_state=42)
        m_val, m_test = train_test_split(m_temp, test_size=0.50, random_state=42)
        
        for split, split_files in [("train", m_tr), ("val", m_val), ("test", m_test)]:
            for f in split_files:
                dest = new_dataset_dir / split / family / f.name
                shutil.copy2(f, dest)

    print(f"\n✅ Balanced dataset built at: {new_dataset_dir}")
    print("[*] Swapping malimg_dataset with balanced version...")
    # Backup
    backup_path = CLEAN_DATA_ROOT / "malimg_dataset_old_backup"
    if backup_path.exists():
        shutil.rmtree(backup_path)
    shutil.move(MALWARE_DATA_DIR, backup_path)
    shutil.move(new_dataset_dir, MALWARE_DATA_DIR)
    print("🎉 Dataset rebuilding complete!")

if __name__ == "__main__":
    main()
