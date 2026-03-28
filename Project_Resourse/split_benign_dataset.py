import os
import shutil
import random
import math

from config import BENIGN_IMAGES_NATARAJ_V3_DIR_STR, MALIMG_ARCHIVE_DIR_STR

# === CONFIGURATION ===
SOURCE_BENIGN_DIR = BENIGN_IMAGES_NATARAJ_V3_DIR_STR
MALIMG_ARCHIVE_DIR = MALIMG_ARCHIVE_DIR_STR

# Splitting Ratios (Must add up to 1.0)
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
# =====================

def main():
    print("🚀 Starting Benign Dataset Splitting (80/10/10)")
    print("-" * 50)
    
    if not os.path.exists(SOURCE_BENIGN_DIR):
        print(f"❌ Source directory not found: {SOURCE_BENIGN_DIR}")
        return
        
    if not os.path.exists(MALIMG_ARCHIVE_DIR):
        print(f"❌ Malimg archive directory not found: {MALIMG_ARCHIVE_DIR}")
        return

    # 1. Gather all benign images
    all_files = [f for f in os.listdir(SOURCE_BENIGN_DIR) if f.endswith('.png')]
    total_files = len(all_files)
    
    print(f"📂 Found {total_files} benign images.")
    
    if total_files == 0:
        print("❌ No images found to split!")
        return

    # 2. Shuffle randomly to ensure an unbiased distribution
    print("🔀 Shuffling files randomly...")
    random.seed(42)  # Seed for reproducibility
    random.shuffle(all_files)

    # 3. Calculate splits
    train_count = math.floor(total_files * TRAIN_RATIO)
    val_count = math.floor(total_files * VAL_RATIO)
    # Give any rounding remainder to the test set
    test_count = total_files - train_count - val_count 

    print(f"📊 Calculated Split (80/10/10):")
    print(f"   Train: {train_count} images")
    print(f"   Val:   {val_count} images")
    print(f"   Test:  {test_count} images")
    print(f"   Total should equal {total_files}: {train_count + val_count + test_count == total_files}")
    
    # 4. Create Target Directories
    target_dirs = {
        "train": os.path.join(MALIMG_ARCHIVE_DIR, "train", "benign"),
        "val": os.path.join(MALIMG_ARCHIVE_DIR, "val", "benign"),
        "test": os.path.join(MALIMG_ARCHIVE_DIR, "test", "benign")
    }
    
    for split_name, dir_path in target_dirs.items():
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 Created/Verified output directory: {dir_path}")

    # 5. Distribute files
    print("\n🚚 Moving files to target directories... (This may take a minute)")
    
    def copy_files(file_list, target_dir, split_name):
        count = 0
        for f in file_list:
            src = os.path.join(SOURCE_BENIGN_DIR, f)
            dst = os.path.join(target_dir, f)
            # Use shutil.copy2 to preserve metadata. 
            # If you want to MOVE instead of copy to save space, use shutil.move(src, dst)
            # Copy is safer just in case something goes wrong.
            shutil.copy2(src, dst)
            count += 1
            if count % 1000 == 0:
                print(f"   [{split_name.upper()}] Copied {count}/{len(file_list)}...")
        print(f"✅ Finished copying {count} files to {split_name}/benign.")

    # Slice the shuffled list
    train_files = all_files[:train_count]
    val_files = all_files[train_count:train_count+val_count]
    test_files = all_files[train_count+val_count:]

    # Execute copies
    copy_files(train_files, target_dirs["train"], "train")
    copy_files(val_files, target_dirs["val"], "val")
    copy_files(test_files, target_dirs["test"], "test")

    print("-" * 50)
    print("🎉 Dataset Integration Complete!")
    print("Your Malimg folder structure now officially contains the 'benign' class spread across train/val/test splits.")
    print("Ready for Stage 2 (CNN Training).")

if __name__ == "__main__":
    main()
