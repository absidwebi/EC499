"""
convert_to_malimg.py  —  EC499 Stage 1 (Revised)
=================================================
Converts benign PE files to grayscale images using the Nataraj width-table
method — the SAME method used to produce the Malimg malware dataset.

WHY THIS MATTERS
----------------
The original version of this script always produced 256×256 images by
zero-padding or truncating every file to 65,536 bytes. Small PE files
(< 32 KB) ended up mostly black, while Malimg malware images are dense
byte textures. This "source bias" gave the CNN a trivial shortcut:
    black lower region → benign
    dense texture throughout → malware

FIX: Apply the Nataraj Width Table
-----------------------------------
Image width is determined by file size — exactly as Malimg was built.

    < 10 KB   →  width = 32
    10–30 KB  →  width = 64
    30–60 KB  →  width = 128
    60–100 KB →  width = 256
   100–200 KB →  width = 384
   200–500 KB →  width = 512
    > 500 KB  →  width = 768

Height = ceil(file_size / width).
The result is a variable-size image — identical in structure to a Malimg image.

dataset_loader.py already NEAREST-resizes both classes to 256×256, so both
benign and malware images pass through the same final standardisation step.
The texture density of both classes now overlaps naturally, removing the shortcut.

USAGE
-----
  # Full conversion (reads from benign_pe_files/, writes to benign_images_nataraj/)
  python convert_to_malimg.py

  # Verify pixel density parity between new benign images and Malimg malware images
  python convert_to_malimg.py --verify
"""

import os
import sys
import math
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from collections import Counter

# ---------------------------------------------------------------------------
# Config import — falls back gracefully if run standalone
# ---------------------------------------------------------------------------
try:
    from config import (
        BENIGN_PE_DIR_STR,
        MALIMG_ARCHIVE_DIR_STR,
        PROJECT_ROOT,
    )
    # New output directory for Nataraj-converted images
    _PROJECT_ROOT = Path(PROJECT_ROOT)
    BENIGN_IMAGES_NATARAJ_DIR_STR = str(_PROJECT_ROOT / "benign_images_nataraj")
except ImportError:
    # Fallback: assume script is run from Project_Resourse/
    _HERE = Path(__file__).resolve().parent
    BENIGN_PE_DIR_STR          = str(_HERE / "benign_pe_files")
    BENIGN_IMAGES_NATARAJ_DIR_STR = str(_HERE / "benign_images_nataraj")
    MALIMG_ARCHIVE_DIR_STR     = str(_HERE / "archive" / "malimg_dataset")

# ---------------------------------------------------------------------------
# Nataraj Width Table  (from Nataraj et al., 2011)
# ---------------------------------------------------------------------------
# Each tuple is (max_file_size_in_bytes, image_width_in_pixels).
# The list is checked top-to-bottom; the first matching upper bound wins.
# Files larger than 500 KB are assigned width 768.
NATARAJ_WIDTH_TABLE = [
    (10   * 1024,  32),   # < 10 KB
    (30   * 1024,  64),   # 10 – 30 KB
    (60   * 1024, 128),   # 30 – 60 KB
    (100  * 1024, 256),   # 60 – 100 KB
    (200  * 1024, 384),   # 100 – 200 KB
    (500  * 1024, 512),   # 200 – 500 KB
]
NATARAJ_DEFAULT_WIDTH = 768  # > 500 KB

# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
MIN_FILE_SIZE_BYTES = 1 * 1024       # Skip files smaller than 1 KB (empty / stub)
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # Skip files larger than 10 MB (edge case)

SUPPORTED_EXTENSIONS = {".exe", ".dll", ".sys", ".ocx"}


# ---------------------------------------------------------------------------
# Core conversion logic
# ---------------------------------------------------------------------------

def get_nataraj_width(file_size_bytes: int) -> int:
    """Return the image width for a given file size using the Nataraj table."""
    for max_size, width in NATARAJ_WIDTH_TABLE:
        if file_size_bytes < max_size:
            return width
    return NATARAJ_DEFAULT_WIDTH


def pe_to_nataraj_image(pe_path: str, output_path: str) -> tuple[bool, int]:
    """
    Convert a single PE file to a variable-size Nataraj-compatible grayscale
    image and save it as a PNG.

    Returns:
        (success: bool, width_used: int)
        width_used is 0 on failure, useful for building the distribution report.
    """
    try:
        file_size = os.path.getsize(pe_path)

        # --- Guards ---
        if file_size < MIN_FILE_SIZE_BYTES:
            return False, 0   # Too small — would produce a near-empty image
        if file_size > MAX_FILE_SIZE_BYTES:
            return False, 0   # Too large — outlier; first 10 MB would dominate

        # --- Determine width and height using the Nataraj table ---
        width  = get_nataraj_width(file_size)
        height = math.ceil(file_size / width)

        # --- Read raw bytes ---
        with open(pe_path, "rb") as f:
            raw_bytes = np.frombuffer(f.read(), dtype=np.uint8)

        # --- Pad to exactly width × height bytes so reshape never fails ---
        total_pixels = width * height
        if len(raw_bytes) < total_pixels:
            raw_bytes = np.pad(raw_bytes, (0, total_pixels - len(raw_bytes)), "constant")
        else:
            raw_bytes = raw_bytes[:total_pixels]

        # --- Reshape to 2D (height, width) — same as Nataraj ---
        image_arr = raw_bytes.reshape((height, width))

        # --- Save as 8-bit grayscale PNG (mode 'L') ---
        img = Image.fromarray(image_arr, mode="L")
        img.save(output_path)
        return True, width

    except Exception as e:
        print(f"  ❌ Error converting {os.path.basename(pe_path)}: {e}")
        return False, 0


# ---------------------------------------------------------------------------
# Main conversion routine
# ---------------------------------------------------------------------------

def run_conversion(source_dir: str, dest_dir: str) -> None:
    """Scan source_dir for PE files and convert all valid ones to dest_dir."""

    source_path = Path(source_dir)
    dest_path   = Path(dest_dir)

    if not source_path.exists():
        print(f"❌ Source directory not found: {source_dir}")
        print("   Make sure benign_pe_files/ exists and contains PE files.")
        sys.exit(1)

    dest_path.mkdir(parents=True, exist_ok=True)

    # Collect all PE files (all supported extensions)
    all_pe_files = [
        f for f in source_path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    total_found = len(all_pe_files)

    if total_found == 0:
        print(f"❌ No PE files found in {source_dir}")
        print(f"   Supported extensions: {SUPPORTED_EXTENSIONS}")
        sys.exit(1)

    print("=" * 60)
    print("  EC499 — Nataraj-Faithful Benign Image Conversion")
    print("=" * 60)
    print(f"  Source : {source_dir}")
    print(f"  Output : {dest_dir}")
    print(f"  Found  : {total_found} PE files")
    print(f"  Method : Nataraj width table (variable-size images)")
    print("-" * 60)

    converted   = 0
    skipped     = 0
    width_counts: Counter = Counter()

    for i, pe_file in enumerate(all_pe_files):
        # Build output filename — preserve stem, force .png extension
        out_filename = pe_file.stem + ".png"
        out_path     = dest_path / out_filename

        success, width_used = pe_to_nataraj_image(str(pe_file), str(out_path))

        if success:
            converted += 1
            width_counts[width_used] += 1
        else:
            skipped += 1

        # Progress report every 500 files
        if (i + 1) % 500 == 0:
            print(f"  Progress: {i + 1}/{total_found} files processed "
                  f"({converted} converted, {skipped} skipped)...")

    # --- Summary ---
    print("-" * 60)
    print(f"  ✅ Conversion complete!")
    print(f"  Total PE files scanned : {total_found}")
    print(f"  Successfully converted : {converted}")
    print(f"  Skipped (size guards)  : {skipped}")
    print(f"  Output directory       : {dest_dir}")
    print()

    # --- Width Distribution Report ---
    # This is the key diagnostic. A healthy distribution has samples across
    # multiple widths (not 90%+ at width=32, which would indicate the PE
    # collection is dominated by tiny files and the bias likely persists).
    print("  📊 Width Distribution Report (Nataraj Table)")
    print("  " + "-" * 44)
    width_labels = {
        32:  "< 10 KB  ",
        64:  "10–30 KB ",
        128: "30–60 KB ",
        256: "60–100 KB",
        384: "100–200 KB",
        512: "200–500 KB",
        768: "> 500 KB ",
    }
    for w in sorted(width_counts.keys()):
        count   = width_counts[w]
        pct     = 100.0 * count / converted if converted > 0 else 0
        bar     = "█" * int(pct / 2)
        label   = width_labels.get(w, f"width={w}")
        print(f"  Width {w:3d}px ({label}): {count:5d} ({pct:5.1f}%) {bar}")
    print()

    # --- Guidance ---
    if width_counts.get(32, 0) / max(converted, 1) > 0.6:
        print("  ⚠️  WARNING: More than 60% of your benign images are width=32")
        print("     (files < 10 KB). These images will still have large padded")
        print("     regions after NEAREST resize to 256×256. Consider recollecting")
        print("     from Program Files or larger software installers to increase")
        print("     the proportion of 64px–256px width images.")
    else:
        print("  ✅ Width distribution looks healthy. Proceed to --verify check.")


# ---------------------------------------------------------------------------
# Pixel density verification routine
# ---------------------------------------------------------------------------

def run_verify(benign_dir: str, malimg_dir: str, n_samples: int = 200) -> None:
    """
    Sample n_samples images from each class and compare mean pixel intensities.
    The goal is to confirm that the source bias has been eliminated.

    Success criterion:
        |benign_mean - malware_mean| < 30 pixel units  (on 0–255 scale)
    """
    import random

    print("=" * 60)
    print("  EC499 — Pixel Density Parity Verification")
    print("=" * 60)

    benign_path  = Path(benign_dir)
    malimg_path  = Path(malimg_dir)

    if not benign_path.exists():
        print(f"❌ Benign image directory not found: {benign_dir}")
        print("   Run conversion first: python convert_to_malimg.py")
        sys.exit(1)

    # Collect benign image paths
    benign_imgs = list(benign_path.glob("*.png"))
    if len(benign_imgs) == 0:
        print("❌ No benign PNG images found. Run conversion first.")
        sys.exit(1)

    # Collect malware image paths from all family subdirectories
    malware_imgs = []
    for family_dir in malimg_path.iterdir():
        if family_dir.is_dir() and family_dir.name != "benign":
            malware_imgs.extend(family_dir.glob("*.png"))

    if len(malware_imgs) == 0:
        print(f"❌ No malware images found in {malimg_dir}")
        sys.exit(1)

    # Sample
    n_b = min(n_samples, len(benign_imgs))
    n_m = min(n_samples, len(malware_imgs))
    sampled_benign  = random.sample(benign_imgs,  n_b)
    sampled_malware = random.sample(malware_imgs, n_m)

    def image_stats(paths):
        means, stds = [], []
        for p in paths:
            try:
                arr = np.array(Image.open(p).convert("L"))
                means.append(float(arr.mean()))
                stds.append(float(arr.std()))
            except Exception:
                pass
        return np.mean(means), np.mean(stds)

    print(f"  Sampling {n_b} benign images  from: {benign_dir}")
    print(f"  Sampling {n_m} malware images from: {malimg_dir} (train split)")
    print()

    b_mean, b_std = image_stats(sampled_benign)
    m_mean, m_std = image_stats(sampled_malware)
    gap           = abs(b_mean - m_mean)

    print(f"  Benign  — Mean pixel intensity: {b_mean:6.2f}  Std: {b_std:6.2f}")
    print(f"  Malware — Mean pixel intensity: {m_mean:6.2f}  Std: {m_std:6.2f}")
    print(f"  Gap (|benign_mean - malware_mean|): {gap:.2f}")
    print()

    if gap < 30:
        print("  ✅ PASS — Pixel density distributions are compatible.")
        print("     Both classes share similar mean brightness. The source bias")
        print("     has been eliminated. You may proceed to split_benign_dataset.py")
    elif gap < 60:
        print("  ⚠️  MARGINAL — Gap is noticeable but may be acceptable.")
        print("     Consider filtering out PE files under 10 KB from your collection")
        print("     and re-running conversion to further reduce the gap.")
    else:
        print("  ❌ FAIL — Gap > 60. Source bias likely persists.")
        print("     The benign images are still much darker than malware images.")
        print("     Action required: recollect more large benign PE files from")
        print("     Program Files, Visual Studio, or similar large software.")
        print("     Then re-run: python convert_to_malimg.py")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert benign PE files to Nataraj-style grayscale images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Run pixel density parity check instead of conversion. "
            "Requires the conversion to have been run first."
        ),
    )
    parser.add_argument(
        "--source",
        default=BENIGN_PE_DIR_STR,
        help="Path to directory containing raw PE files (default: benign_pe_files/)",
    )
    parser.add_argument(
        "--output",
        default=BENIGN_IMAGES_NATARAJ_DIR_STR,
        help="Path to write converted PNG images (default: benign_images_nataraj/)",
    )
    parser.add_argument(
        "--malimg-train",
        default=str(Path(MALIMG_ARCHIVE_DIR_STR) / "train"),
        help="Path to Malimg train split for --verify comparison.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of images to sample per class during --verify (default: 200)",
    )
    args = parser.parse_args()

    if args.verify:
        run_verify(
            benign_dir=args.output,
            malimg_dir=args.malimg_train,
            n_samples=args.samples,
        )
    else:
        run_conversion(source_dir=args.source, dest_dir=args.output)


if __name__ == "__main__":
    main()
