# PROJECT_CONTEXT.md
# EC499 — Adversarial Robustness in Deep Learning-Based Malware Detection

**Student:** Abdulsalam Ashraf Aldwebi (ID: 2210245306)
**Supervisor:** Dr. Suad Elgeder | University of Tripoli
**GitHub:** https://github.com/absidwebi/EC499
**Primary Machine:** Ubuntu PC, RTX 4060, `/home/alucard-00/EC499/`
**Python Environment:** `/home/alucard-00/EC499/Project_Resourse/venv/` (Python 3.10.12)

---

## 1. Problem Definition

Deep learning malware detectors are vulnerable to adversarial attacks — imperceptible perturbations to malicious PE files that cause misclassification as benign while preserving malicious functionality. This project builds, attacks, and defends an image-based malware classifier.

**Binary classification task:** Benign (0) vs Malware (1)

---

## 2. Approach — PE File to Grayscale Image

Raw PE binary files are read byte-by-byte. Each byte (0–255) maps directly to a pixel grayscale value. The byte array is reshaped into a 2D image using the Nataraj (2011) variable-width method.

**Nataraj Width Table:**

| File Size | Image Width |
|---|---|
| < 10 KB | 32 px |
| 10–30 KB | 64 px |
| 30–60 KB | 128 px |
| 60–100 KB | 256 px |
| 100–200 KB | 384 px |
| 200–500 KB | 512 px |
| 500 KB–1 MB | 768 px |
| > 1 MB | 1024 px |

Height = ceil(file_size / width). Final image is zero-padded to fill the last row if needed.

**CRITICAL:** NO interpolation is ever used. Bilinear/bicubic resizing blurs byte-level textures and creates structural artifacts that models exploit as shortcuts.

---

## 3. Dataset Description

### 3.1 Malware Dataset — Malimg (Kaggle)

- 9,339 total images across 25 malware families
- Already split into train/val/test by Kaggle
- Dominant family: Allaple.A (2,949 images, 31.6% of dataset) — large worm, 512–768px wide images
- After cross-split deduplication: Yuner.A has 640 train samples, 0 val/test
- 669 internal train duplicates (known Malimg property)

**Final split sizes after deduplication and benign integration:**
**Final split sizes (CURRENT v3 benign integrated):**
- Train: 22,288 (14,829 benign + 7,459 malware)
- Val: 2,663 (1,853 benign + 810 malware)
- Test: 2,702 (1,855 benign + 847 malware)

### 3.2 Benign Dataset

**Original (BIASED — replaced):** ~13,147 images from Windows System32/SysWOW64. Problem: mostly < 60 KB files producing sparse 32px/64px images. Creates structural separation from dense malware images.

**New v3 (CURRENT):** 18,541 PE files from:
- PortableApps (12,200+ files, 80+ different applications)
- Program Files: MATLAB, Altium, Microsoft Office, PowerShell, Git, Proton, KiCad, NVIDIA (each capped at 500 files via per-vendor limit)
- Size filter: 50 KB minimum, 200 MB maximum
- Width distribution: 0% at 32/64px, 5.6% at 128px, 14.6% at 256px, 19.3% at 384px, 23.5% at 512px, 13.2% at 768px, 23.8% at 1024px

**Location on Windows:** `C:\Users\الصدارة\Desktop\EC499\benign_images_nataraj_v3\`
**Target on Ubuntu:** `/home/alucard-00/EC499/benign_images_nataraj_v3/`

**Status (CURRENT):** Transferred to Ubuntu and integrated into `Project_Resourse/archive/malimg_dataset/*/benign/`.

### 3.3 Dataset Folder Structure on Ubuntu

```
Project_Resourse/archive/malimg_dataset/
├── train/   (26 folders: 25 malware families + benign)
├── val/     (25 folders: Yuner.A removed, benign present)
└── test/    (25 folders: Yuner.A removed, benign present)
```

---

## 4. Preprocessing Pipeline

### 4.1 dataset_loader.py — PadTo256 Transform

`PadTo256` preserves byte structure by **crop/pad only** (no interpolation):

```python
class PadTo256:
    def __call__(self, img):
        w = img.width
        h = img.height
        target = 256
        if w > target or h > target:
            img = F.center_crop(img, [target, target])
        if w < target or h < target:
            img = F.pad(img, [0, 0, target-w, target-h], fill=0)
        return img
```

**Full transform pipeline:**
```python
transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    PadTo256(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # → tensors in [-1, 1]
])
```

**Padding shortcut mitigation (CURRENT work):** `dataset_loader.py` supports an optional post-normalization transform that randomizes ONLY the padded bottom-right rectangle:
- Enable via env: `PAD_NOISE=1` (optional `PAD_NOISE_MODE=uniform|normal`)
- Purpose: prevent the model from using the “all -1 padded rows/cols” layout as a shortcut.

### 4.2 Binary Label Mapping

ImageFolder assigns integer indices alphabetically. "benign" sits between malware families. A per-split target_transform remaps:
- benign folder index → 0
- all other indices → 1

**CRITICAL:** Each split computes its OWN benign index. Yuner.A removal causes benign to shift from index 25 (train) to 24 (val/test).

```python
idx = split_ds.classes.index('benign')
split_ds.target_transform = lambda x, i=idx: 0 if x == i else 1
```

### 4.3 Class Weights

```python
weight_for_0 = total_samples / (2.0 * benign_count)
weight_for_1 = total_samples / (2.0 * malware_count)
pos_weight = class_weights[1] / class_weights[0]  # passed to BCEWithLogitsLoss
```

---

## 5. Model Architectures

### 5.1 ResNet-18 (Primary Model)

```python
def get_resnet18_grayscale():
    model = models.resnet18(weights=None)
    # Modify first conv: 3 channels → 1 channel
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Modify classifier head
    model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(512, 1))
    return model
```

- Parameters: 11,180,373
- Input: [B, 1, 256, 256]
- Output: [B, 1] (single logit)
- Weights: `models/resnet18_clean_vulnerable.pth`
- AT weights: `models/resnet18_adversarially_trained.pth`

### 5.2 EfficientNet-B0 (Comparison Model)

```python
def get_efficientnet_b0_grayscale():
    model = tv_models.efficientnet_b0(weights=None)
    model.features[0][0] = nn.Conv2d(1, 32, ...)  # 3→1 channel
    model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(1280, 1))
    return model
```

- Parameters: 4,008,253 (2.79× fewer than ResNet-18)
- Same input/output format
- Weights: `models/efficientnet_b0_clean_vulnerable.pth`

### 5.3 CustomCNN (Legacy — superseded by ResNet-18)

4-block CNN: Conv→BN→ReLU→MaxPool × 4, then Flatten→FC(512)→Dropout(0.5)→FC(1)
Weights: `models/custom_cnn_clean_vulnerable.pth` (kept for reference only)

---

## 6. Training Pipeline

**Loss function:** BCEWithLogitsLoss with pos_weight
**Optimizer:** AdamW, LR=1e-4
**Scheduler:** ReduceLROnPlateau (factor=0.5, patience=3)
**Batch size:** 32 (clean training), 16 (adversarial training)
**Epochs:** 20 with early stopping (patience=5)
**Seed:** torch.manual_seed(42)
**num_workers:** 0 (prevents CUDA fork OOM on Linux)

**Key constraint:** NO geometric augmentations (rotations, flips, crops destroy PE byte structure).

---

## 7. Attack Implementations

All attacks are manually implemented using BCEWithLogitsLoss (not torchattacks library, which uses CrossEntropyLoss incompatible with binary single-logit output).

**Clamp range: [-1.0, 1.0] throughout (NOT [0, 1])**

```python
# FGSM
perturbation = eps * images.grad.sign()
adv = torch.clamp(images + perturbation, -1.0, 1.0)

# PGD (Madry et al. 2018)
adv = images + uniform(-eps, eps)  # random start
for _ in range(steps):
    adv = adv + alpha * grad.sign()
    delta = clamp(adv - images, -eps, eps)
    adv = clamp(images + delta, -1.0, 1.0)
```

**Adversarial training config:** ε=0.05, α=0.01, 7 steps per batch, warm-started from clean model.

---

## 8. Evaluation Metrics

---

## 9. MaleX Integration Update (2026-03-29)

The pipeline has pivoted to MaleX byteplot images for the active training/robustness track while keeping Malimg assets for comparison.

### Active MaleX dataset roots

- Source benign: `/home/alucard-00/EC499/ben_byteplot_imgs_zipped/byteplot_imgs_RxR/256/`
- Source malware: `/home/alucard-00/EC499/mal_byteplot_imgs_zipped/byteplot_imgs_RxR/256/`
- Working split root: `/home/alucard-00/EC499/Project_Resourse/archive/malex_dataset/`

### MaleX split sizes

- Train: 287,560 (143,780 benign + 143,780 malware)
- Val: 35,944 (17,972 benign + 17,972 malware)
- Test: 35,946 (17,973 benign + 17,973 malware)

### MaleX clean model training (Stage 2)

- Script: `Project_Resourse/train.py`
- Output model: `Project_Resourse/models/resnet18_malex_clean_vulnerable.pth`
- Training log: `run_logs/train_resnet18_malex_stage2.log`
- Status: completed with early stopping at epoch 7
- Best validation loss: 0.3695 at epoch 2
- Epoch 7 metrics: Train Loss 0.0858, Train Acc 96.91%, Val Loss 0.5757, Val Acc 83.16%

### Stage 3 Part 1 attack evaluation (MaleX clean model)

- Script: `Project_Resourse/evaluate_attacks.py`
- Log: `run_logs/evaluate_attacks_malex_stage3.log`
- Result summary (`Project_Resourse/logs/attack_evaluation_results.txt`):
    - Clean: 82.26%
    - FGSM eps 0.01: 9.88%
    - FGSM eps 0.02: 1.37%
    - FGSM eps 0.05: 0.15%
    - FGSM eps 0.10: 0.00%
    - PGD eps 0.01 steps 10: 5.91%
    - PGD eps 0.02 steps 20: 0.44%
    - PGD eps 0.05 steps 40: 0.00%

Interpretation: clean MaleX model is highly vulnerable and suitable as the warm-start baseline for adversarial defense training.

### Stage 3 Part 2 adversarial training (in progress)

- Script: `Project_Resourse/adversarial_train.py`
- Log: `run_logs/adversarial_train_malex_stage3.log`
- Warm-start checkpoint: `Project_Resourse/models/resnet18_malex_clean_vulnerable.pth`
- Robust output path: `Project_Resourse/models/resnet18_malex_adversarially_trained.pth`
- Current status: active run, epoch 1 in progress (batch-level logging visible)

### Prompt-phase status snapshot (requested reporting format)

| Phase | Status | Notes |
|---|---|---|
| Phase 1 | PASS | MaleX source validation passed (sizes/modes/corruption checks) |
| Phase 2 | PASS | Balanced MaleX split created under `archive/malex_dataset` |
| Phase 3 | PASS | `config.py` MaleX constants added; import verification passed |
| Phase 4 | PASS | `dataset_loader.py` replaced for MaleX and self-test passed |
| Phase 5 | PASS | `train.py`, `evaluate_attacks.py`, `adversarial_train.py` rewired to MaleX |
| Phase 6A | FAIL | Hash overlap check detected cross-split overlaps |
| Phase 6B | PASS | Label/tensor/range checks passed |
| Phase 6C | PASS | Shuffle sanity check stayed near chance-level validation |
| Phase 7 | PASS | Clean Stage 2 MaleX training completed with early stopping |
| Stage 3 Part 1 | PASS | Attack vulnerability evaluation completed |
| Stage 3 Part 2 | RUNNING | Adversarial defense training active |

- Clean accuracy (combined)
- Split-class: malware recall, benign correct rate
- Evasion rate = 100 - malware recall under attack
- Accuracy vs epsilon curve
- Black-box transfer attack (gradient masking detection)
- Logit distribution by class
- Input gradient saliency maps

**Bias/shortcut indicators (CURRENT):**
- Padding-only LR AUC (from normalized tensors)
- Logit gap mean (benign vs malware). Large gaps (e.g. ~22 units) indicate shortcut-driven separation.

---

## 9. Fixed Adversarial Test Set

Location: `Project_Resourse/adversarial_test_set/`
- `fgsm_eps0.05/images/` — 847 adversarial PNGs
- `pgd_eps0.05_steps40/images/` — 847 adversarial PNGs

Generated from malware-only test samples using clean ResNet-18. Both clean and AT models are evaluated on the identical pixel values for fair comparison.

---

## 10. Codebase Reference

| Script | Purpose |
|---|---|
| `config.py` | All path constants |
| `dataset_loader.py` | DataLoaders, PadTo256, binary labels, class weights |
| `models.py` | All model definitions |
| `train.py` | ResNet-18 clean training |
| `train_efficientnet.py` | EfficientNet-B0 clean training |
| `adversarial_train.py` | ResNet-18 adversarial training |
| `adversarial_train_efficientnet.py` | EfficientNet-B0 adversarial training |
| `evaluate_attacks.py` | FGSM/PGD evaluation |
| `generate_adversarial_test_set.py` | Fixed adversarial test set generator |
| `split_benign_dataset.py` | Places benign images into Malimg folder structure |
| `check_hash_overlaps.py` | Cross-split deduplication check |
**Shortcut mitigation (CURRENT experiments):** `train.py` supports an optional training-only `WeightedRandomSampler` driven by a padding-only logistic regression on features `(pad_rows_all_-1, pad_cols_all_-1, frac_-1_pixels)`.
- Enable via env: `PAD_NEUTRALIZE_SAMPLER=1`
- Optional: `DATA_DIR_OVERRIDE=...`, `PAD_NEUTRALIZE_WEIGHT_CLIP_MAX=...`
