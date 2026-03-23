# Dataset Preparation & Strategy Guide

This report addresses your questions regarding folder structures, label encoding, and dataset balancing before moving into Stage 2 (CNN Training).

---

## 1. Folder Structure & Splitting
> **Question:** *The Malimg folder structure I downloaded from Kaggle contains train, val, and test splits in different folders. How should I combine them?*

### Standard PyTorch `ImageFolder` Approach
To train a CNN in PyTorch, the easiest and most standard method is to use `torchvision.datasets.ImageFolder`. This requires a specific directory hierarchy where **each class has its own subfolder**.

Since your Malimg dataset is already split into `train/`, `val/`, and `test/`, you should physically integrate your `benign` images into this existing structure.

**Target Structure:**
```text
dataset/
├── train/
│   ├── malware_family_1/
│   ├── malware_family_2/
│   ├── ...
│   └── benign/                 <-- Put 70% of benign images here
├── val/
│   ├── malware_family_1/
│   ├── ...
│   └── benign/                 <-- Put 15% of benign images here
└── test/
    ├── malware_family_1/
    ├── ...
    └── benign/                 <-- Put 15% of benign images here
```

**Next Step:** We will need to write a quick PyTorch/Python script to take your `benign_images_256x256` folder, shuffle the files, and move them into the `train/benign/`, `val/benign/`, and `test/benign/` folders matching the ratios of the Malimg dataset.

---

## 2. Labeling the Dataset
> **Question:** *Should we label the dataset (benign and malware image folders) with the right corresponding label so the network learns the output?*

**Answer: Yes, but the folders DO the labeling for you!**

If you use the folder structure outlined above, you **do not** need to manually rename files or create complex CSV files. In PyTorch:
*   The `ImageFolder` class automatically assigns labels based on the subfolder names.
*   For example, it will assign `benign = 0`, `Adialer.C = 1`, `Agent.FYI = 2`, etc.
*   If you are doing **Binary Classification** (Malware vs. Benign), we will modify the PyTorch `Dataset` class to map all 25 malware folders to `1` and the benign folder to `0`.
*   If you are doing **Multi-class Classification** (predicting the exact family *or* benign), it will use the 26 unique labels (25 families + 1 benign).

---

## 3. Dataset Balancing: 50/50 vs. "Use Everything"
> **Question:** *What do you suggest we use for the dataset split? 50/50 for benign and malware images during training, or should we use a higher number of samples for the benign images (15k vs 9k)?*

This is the most critical question for Stage 2. 

**Total counts:**
*   Malimg (Malware): ~9,339 images
*   Benign: 15,000 images

### Option A: The 50/50 Split (Recommended for Baseline)
*   **What it is:** Downsample your 15k benign images to precisely match the ~9.3k limit of the malware set (yielding ~18,600 images total).
*   **Pros:** Prevents **class-imbalance bias**. If a neural network sees 60% benign images during training, it might lazily guess "benign" more frequently just to boost its accuracy metric artificially.
*   **Cons:** You throw away ~5,600 perfectly good benign training examples.

### Option B: Use All Data (Weighted Loss) - Highly Recommended
*   **What it is:** Keep all 9.3k malware and all 15k benign images (~24,300 total). 
*   **How we fix the bias:** We tell PyTorch to mathematically penalize the network harder when it misclassifies a malware image to compensate for the imbalance. This is called **Class Weighting** (specifically in Cross-Entropy Loss).
*   **Pros:** The network sees maximum diversity in benign software (very important for adversarial defense later!), but doesn't become biased. Deep Learning thrives on *more data*.

### 🏆 Verdict & Recommendation
**Go with Option B (Use all data, but use Class Weights in PyTorch).**
Since your ultimate goal is "Adversarial Robustness," having a highly diverse baseline understanding of what a "benign" file looks like is crucial. The more varied the benign dataset, the harder it is for an attacker to craft an adversarial perturbation that successfully mimics one.

**Action Plan:**
When we write the PyTorch training loop in Stage 2, we will calculate the ratio (e.g., 9000/15000) and pass a `weight` tensor to the Loss Function.

---

## Next Steps
Before you start coding the CNN, we need to execute the file reorganization logic discussed in Section 1. 

**Shall I write a script to automatically split your 15,000 benign images into the `archive/malimg_dataset/train/benign`, `val/benign`, and `test/benign` folders to complete Stage 1?**
