import os
from PIL import Image
from collections import Counter

BENIGN_SRC  = "/home/alucard-00/EC499/ben_byteplot_imgs_zipped/byteplot_imgs_RxR/256"
MALWARE_SRC = "/home/alucard-00/EC499/mal_byteplot_imgs_zipped/byteplot_imgs_RxR/256"

def verify_folder(folder_path, label):
    print(f"\n--- Verifying {label} ---")
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    print(f"Total PNG files: {len(files)}")

    sizes = Counter()
    modes = Counter()
    corrupt = 0
    sample_checked = 0

    for fname in files:
        fpath = os.path.join(folder_path, fname)
        try:
            with Image.open(fpath) as img:
                img.load()
                sizes[img.size] += 1
                modes[img.mode] += 1
                sample_checked += 1
        except Exception as e:
            corrupt += 1
            if corrupt <= 5:
                print(f"  CORRUPT: {fname} — {e}")

    print(f"Corrupt/unreadable: {corrupt}")
    print(f"Image sizes found: {dict(sizes.most_common(10))}")
    print(f"Image modes found: {dict(modes)}")

    # PASS criteria
    non_256 = sum(v for k, v in sizes.items() if k != (256, 256))
    if non_256 > 0:
        print(f"FAIL: {non_256} images are NOT 256x256")
    else:
        print(f"PASS: All {sample_checked} images are 256x256")

    if corrupt > 0:
        print(f"FAIL: {corrupt} corrupt images found")
    else:
        print(f"PASS: No corrupt images")

    return len(files), corrupt, non_256

benign_count, b_corrupt, b_non256 = verify_folder(BENIGN_SRC, "BENIGN")
malware_count, m_corrupt, m_non256 = verify_folder(MALWARE_SRC, "MALWARE")

print(f"\n=== PHASE 1 SUMMARY ===")
print(f"Benign images:  {benign_count}")
print(f"Malware images: {malware_count}")
print(f"Imbalance ratio: {malware_count / benign_count:.2f}:1")

if b_corrupt + m_corrupt + b_non256 + m_non256 == 0:
    print("PHASE 1: ALL CHECKS PASSED")
else:
    print("PHASE 1: FAILED — DO NOT PROCEED")
