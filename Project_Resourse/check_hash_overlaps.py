import hashlib
import os
from pathlib import Path
from tqdm import tqdm

def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def check_overlaps(data_dir):
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    print(f"[*] Analyzing directories in: {data_dir}")

    results = {}
    for split_name, split_path in [("train", train_dir), ("val", val_dir), ("test", test_dir)]:
        if not split_path.exists():
            print(f"[!] Warning: {split_name} directory not found at {split_path}")
            results[split_name] = {}
            continue
        
        hashes = {}
        files = list(split_path.rglob("*.png"))
        print(f"[*] Hashing {len(files)} files in {split_name}...")
        for p in tqdm(files, desc=split_name):
            h = file_hash(p)
            if h in hashes:
                hashes[h].append(str(p))
            else:
                hashes[h] = [str(p)]
        results[split_name] = hashes

    # Check overlaps
    train_hashes = set(results["train"].keys())
    val_hashes = set(results["val"].keys())
    test_hashes = set(results["test"].keys())

    train_val = train_hashes & val_hashes
    train_test = train_hashes & test_hashes
    val_test = val_hashes & test_hashes

    print("\n" + "="*50)
    print("HASH OVERLAP REPORT")
    print("="*50)
    print(f"Train vs Val overlap:  {len(train_val)} unique hashes")
    print(f"Train vs Test overlap: {len(train_test)} unique hashes")
    print(f"Val vs Test overlap:   {len(val_test)} unique hashes")
    
    if len(train_val) > 0:
        print("\n[!] DATA LEAKAGE DETECTED: Files shared between Train and Val!")
        # Print a few examples
        for h in list(train_val)[:3]:
            print(f"  Hash: {h}")
            print(f"    Train: {results['train'][h][0]}")
            print(f"    Val:   {results['val'][h][0]}")
    
    if len(train_test) > 0:
        print("\n[!] DATA LEAKAGE DETECTED: Files shared between Train and Test!")

    print("="*50)

if __name__ == "__main__":
    from config import MALIMG_ARCHIVE_DIR_STR
    check_overlaps(MALIMG_ARCHIVE_DIR_STR)
