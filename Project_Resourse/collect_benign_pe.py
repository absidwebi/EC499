import os
import shutil
import hashlib
import pefile

def is_valid_pe(file_path):
    """Check if file is a valid PE using pefile."""
    try:
        pe = pefile.PE(file_path, fast_load=True)
        if pe.DOS_HEADER.e_magic != 0x5A4D:
            pe.close()
            return False
        pe.close()
        return True
    except Exception:
        return False

def get_file_hash(file_path):
    """Compute SHA-256 hash of file for deduplication."""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except (PermissionError, OSError):
        return None

def collect_benign_pe(source_dirs, dest_dir, max_files=15000):
    """
    Collect benign PE files from multiple source directories to dest_dir.
    Resumes from existing collection state.
    """
    os.makedirs(dest_dir, exist_ok=True)
    
    seen_hashes = set()
    collected = 0
    
    print("Initializing: Scanning existing dataset for duplicates...")
    # 1. Scan existing files in destination to populate seen_hashes and set index
    existing_files = [f for f in os.listdir(dest_dir) if f.startswith("benign_") and f.endswith(".exe")]
    
    # Set collection counter based on highest existing index to verify resume
    max_index = 0
    for f in existing_files:
        try:
            # Extract number from benign_XXXXX.exe
            idx = int(f.split('_')[1].split('.')[0])
            if idx > max_index:
                max_index = idx
        except (IndexError, ValueError):
            pass
            
    collected = len(existing_files) # Start count from actual number of files
    next_index = max_index + 1      # Start naming from next available index
    
    print(f"Found {collected} existing files. Resuming collection from index {next_index}...")
    
    # Optional: If you want strict deduplication against existing files, uncomment below.
    # It reads every existing file which takes time (approx 1-2 mins for 5GB).
    # For now, we assume existing files are unique and valid. 
    # To be safe against adding SAME file from SysWOW64 that is in System32:
    print("Hashing existing files to prevent duplicates (this might take a minute)...")
    for f in existing_files:
        full_path = os.path.join(dest_dir, f)
        h = get_file_hash(full_path)
        if h:
            seen_hashes.add(h)
    print(f"Indexed {len(seen_hashes)} unique hashes from existing dataset.")
    print("-" * 50)

    for source_dir in source_dirs:
        if collected >= max_files:
            break
            
        print(f"Scanning {source_dir}...")
        
        if not os.path.exists(source_dir):
            print(f"Warning: {source_dir} not found. Skipping.")
            continue

        for root, _, files in os.walk(source_dir):
            for filename in files:
                if collected >= max_files:
                    break
                
                if not filename.lower().endswith(('.exe', '.dll', '.sys', '.ocx')):
                    continue

                src_path = os.path.join(root, filename)
                
                try:
                    file_size = os.path.getsize(src_path)
                    if file_size < 1024 or file_size > 50_000_000:
                        continue
                except OSError:
                    continue

                if not is_valid_pe(src_path):
                    continue

                file_hash = get_file_hash(src_path)
                if not file_hash or file_hash in seen_hashes:
                    continue
                
                seen_hashes.add(file_hash)

                # Copy to destination with incrementing index
                dest_path = os.path.join(dest_dir, f"benign_{next_index:05d}.exe")
                
                try:
                    shutil.copy2(src_path, dest_path)
                    collected += 1
                    next_index += 1
                    
                    if collected % 100 == 0:
                        print(f"Total Collected: {collected} | Added: {filename}")
                except (PermissionError, OSError):
                    continue
            
            if collected >= max_files:
                break

    print("-" * 50)
    print(f"✅ Collection Complete!")
    print(f"Total Valid PE Files: {collected}")
    print(f"Location: {dest_dir}")

from config import BENIGN_SOURCE_DIRS, BENIGN_PE_DIR_STR

# === CONFIGURATION ===
SOURCE_DIRS = BENIGN_SOURCE_DIRS
DEST_DIR = BENIGN_PE_DIR_STR
# =====================

if __name__ == "__main__":
    collect_benign_pe(SOURCE_DIRS, DEST_DIR, max_files=15000)