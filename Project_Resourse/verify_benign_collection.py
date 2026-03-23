import os
import hashlib
import pefile
import sys

from config import BENIGN_PE_DIR_STR

# === CONFIGURATION ===
DATASET_DIR = BENIGN_PE_DIR_STR
# =====================

def get_file_hash(file_path):
    """Compute SHA-256 hash of file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception:
        return None

def verify_dataset(directory):
    print(f"🔍 Verifying dataset in: {directory}")
    
    if not os.path.exists(directory):
        print(f"❌ Error: Directory not found!")
        return

    files = [f for f in os.listdir(directory) if f.endswith('.exe')]
    total_files = len(files)
    
    print(f"📂 Found {total_files} .exe files. Starting verification...")
    print("-" * 60)

    unique_hashes = set()
    valid_pe_count = 0
    corrupted_count = 0
    duplicate_count = 0
    access_error_count = 0

    for i, filename in enumerate(files):
        file_path = os.path.join(directory, filename)
        
        # progress update every 1000 files
        if (i + 1) % 1000 == 0:
            print(f"Processing {i + 1}/{total_files}...")

        # 1. Check if file is readable
        try:
            with open(file_path, 'rb') as f:
                pass
        except Exception:
            print(f"❌ Access Error: {filename}")
            access_error_count += 1
            continue

        # 2. Check PE Validity
        try:
            pe = pefile.PE(file_path, fast_load=True)
            if pe.DOS_HEADER.e_magic != 0x5A4D:
                 print(f"❌ Invalid Magic Number: {filename}")
                 corrupted_count += 1
                 pe.close()
                 continue
            pe.close()
            valid_pe_count += 1
        except Exception:
            print(f"❌ PE Parse Error: {filename}")
            corrupted_count += 1
            continue

        # 3. Check for Duplicates
        file_hash = get_file_hash(file_path)
        if file_hash in unique_hashes:
            print(f"⚠️ Duplicate Found: {filename} (Hash collision)")
            duplicate_count += 1
        else:
            if file_hash:
                unique_hashes.add(file_hash)

    print("-" * 60)
    print("📊 VERIFICATION RESULTS")
    print("-" * 60)
    print(f"Total Files Scanned: {total_files}")
    print(f"✅ Valid PE Files:     {valid_pe_count}")
    print(f"❌ Corrupted/Invalid:  {corrupted_count}")
    print(f"⚠️ Duplicates:         {duplicate_count}")
    print(f"🚫 Access Errors:      {access_error_count}")
    print("-" * 60)
    
    if valid_pe_count == total_files and duplicate_count == 0:
        print("✅ SUCCESS: Dataset is clean, valid, and unique.")
    else:
        print("⚠️ WARNING: Dataset contains issues. See details above.")

if __name__ == "__main__":
    verify_dataset(DATASET_DIR)
