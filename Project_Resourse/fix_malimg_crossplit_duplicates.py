"""
fix_malimg_crosssplit_duplicates.py

Removes cross-split duplicate images from the Malimg malware families only.
Strategy:
  - Train is the authority. Its hashes are never removed.
  - If a val or test image has the same hash as any train image
    in the same family, the val/test copy is removed.
  - If a test image has the same hash as a val image (but not train),
    the test copy is removed.
  - Benign folders are NEVER touched under any circumstance.
  - Deduplication is per-family only — hashes are never compared
    across different malware families to avoid false-duplicate removal.
"""

import os
import hashlib
import shutil
from pathlib import Path
from collections import defaultdict


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def fix_crosssplit_duplicates(dataset_dir):
    dataset_dir = Path(dataset_dir)
    print(f"[*] Targeted cross-split deduplication for: {dataset_dir}")
    print("[*] Strategy: train is authority. Benign folders never touched.")
    print("-" * 60)

    total_removed = 0
    families_affected = 0

    # Get all malware family names from train (excludes benign)
    train_path = dataset_dir / "train"
    families = [
        d for d in os.listdir(train_path)
        if os.path.isdir(train_path / d) and d != "benign"
    ]

    for family in sorted(families):
        # Build hash set for this family's train images (authority)
        train_family_path = dataset_dir / "train" / family
        train_hashes = set()
        for img in train_family_path.glob("*.png"):
            train_hashes.add(file_hash(img))

        family_removed = 0

        # Check val — remove if hash exists in train
        val_family_path = dataset_dir / "val" / family
        val_hashes = set()
        if val_family_path.exists():
            for img in list(val_family_path.glob("*.png")):
                h = file_hash(img)
                if h in train_hashes:
                    img.unlink()
                    family_removed += 1
                    total_removed += 1
                else:
                    val_hashes.add(h)

        # Check test — remove if hash exists in train OR val
        test_family_path = dataset_dir / "test" / family
        if test_family_path.exists():
            for img in list(test_family_path.glob("*.png")):
                h = file_hash(img)
                if h in train_hashes or h in val_hashes:
                    img.unlink()
                    family_removed += 1
                    total_removed += 1

        if family_removed > 0:
            families_affected += 1
            print(f"  {family:<25} removed {family_removed} duplicate(s)")

    print("-" * 60)
    print(f"[*] Total duplicates removed: {total_removed}")
    print(f"[*] Families affected: {families_affected}")
    print("[*] Benign folders: untouched")
    print("[OK] Cross-split deduplication complete.")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import MALIMG_ARCHIVE_DIR_STR
    fix_crosssplit_duplicates(MALIMG_ARCHIVE_DIR_STR)
