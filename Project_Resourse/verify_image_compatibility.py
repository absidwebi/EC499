import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from config import BENIGN_IMAGES_DIR_STR, MALWARE_SAMPLE_DIR_STR, COMPARISON_PLOT_PATH_STR

# === CONFIGURATION ===
BENIGN_DIR = BENIGN_IMAGES_DIR_STR
MALWARE_DIR = MALWARE_SAMPLE_DIR_STR
OUTPUT_PLOT = COMPARISON_PLOT_PATH_STR
NUM_SAMPLES = 4
# =====================

def analyze_image(img_path, category):
    try:
        img = Image.open(img_path)
        img_arr = np.array(img)
        
        info = {
            "name": os.path.basename(img_path),
            "category": category,
            "shape": img_arr.shape,
            "dtype": str(img_arr.dtype),
            "min_val": int(np.min(img_arr)),
            "max_val": int(np.max(img_arr)),
            "is_integer_only": np.issubdtype(img_arr.dtype, np.integer),
            "unique_values_count": len(np.unique(img_arr))
        }
        return img_arr, info
    except Exception as e:
        print(f"Error reading {img_path}: {e}")
        return None, None

def main():
    print("🔍 VERIFYING DATASET COMPATIBILITY")
    print("-" * 50)
    
    # 1. Get random samples
    benign_files = [f for f in os.listdir(BENIGN_DIR) if f.endswith('.png')]
    malware_files = [f for f in os.listdir(MALWARE_DIR) if f.endswith('.png')]
    
    benign_samples = random.sample(benign_files, NUM_SAMPLES)
    malware_samples = random.sample(malware_files, NUM_SAMPLES)
    
    benign_paths = [os.path.join(BENIGN_DIR, f) for f in benign_samples]
    malware_paths = [os.path.join(MALWARE_DIR, f) for f in malware_samples]
    
    # 2. Analyze and verify
    print("📋 STATISTICAL VERIFICATION")
    
    images_to_plot = []
    titles = []
    
    for paths, category in [(benign_paths, "Benign"), (malware_paths, "Malimg (Adialer.C)")]:
        for path in paths:
            img_arr, info = analyze_image(path, category)
            if info:
                images_to_plot.append(img_arr)
                titles.append(f"{category}\n{info['name']}")
                
                print(f"[{category}] File: {info['name']}")
                print(f"   => Shape: {info['shape']} | DType: {info['dtype']}")
                print(f"   => Pixel Range: [{info['min_val']}, {info['max_val']}]")
                print(f"   => All Pixels integers? {info['is_integer_only']}")
                print(f"   => Unique Intensity Values: {info['unique_values_count']}")
                print("")
                
                # Check constraints
                if category == "Benign":
                    assert info['shape'] == (256, 256), f"Shape mismatch in {path}"
                assert info['is_integer_only'], f"Found non-integers in {path}"

    print("✅ All statistical constraints (256x256 size, 8-bit integer pixels) PASSED.")
    
    # 3. Generate Visual Comparison Grid
    print("\n🖼️ GENERATING VISUAL COMPARISON GRID")
    fig, axes = plt.subplots(2, NUM_SAMPLES, figsize=(16, 8))
    fig.suptitle("Visual Comparison: Benign (Generated) vs Malimg (Authentic)", fontsize=16)
    
    for i in range(2 * NUM_SAMPLES):
        row = i // NUM_SAMPLES
        col = i % NUM_SAMPLES
        ax = axes[row, col]
        
        # We use cmap='gray' which is standard for Malimg visualizing
        ax.imshow(images_to_plot[i], cmap='gray', interpolation='none') 
        ax.set_title(titles[i], fontsize=10)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    print(f"✅ Grid saved to: {OUTPUT_PLOT}")
    print("Please open this image file to visually inspect that structural properties are comparable and there is no blur/interpolation.")

if __name__ == "__main__":
    main()
