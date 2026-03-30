# CURRENT_STATE.md
# EC499 — Current Project State
**Last updated:** 2026-03-30

---

## 1. Overall Progress

| Stage | Status | Notes |
|---|---|---|
| Stage 1 — Dataset Preparation | ✅ Updated | v3 benign integrated into Malimg folder structure |
| Stage 2 — Model Training | ✅ Re-run on v3 | ResNet-18 trained; results still show shortcut signals |
| Stage 3 — Adversarial Attack & Defense | ⏸️ Paused | Base-model selection blocked by 3C2D plateau and pretrained ResNet overfitting |
| Stage 4 — GitHub + Documentation | ✅ Complete | Repo: https://github.com/absidwebi/EC499 |
| Stage 5 — Dataset Bias Fix | 🔄 In Progress | Now focused on padding-layout shortcut, not System32 size bias |

---

## 2. Current Focus

**Immediate blocker (MaleX track):** both base-model candidates have stalled below target validation quality.

- 3C2D (fixed + resume-enabled) plateaued after epoch 49 despite extension to 70 epochs.
- Pretrained ResNet-18 overfit quickly (train acc increased strongly while val stagnated).
- Pretrained run has been manually stopped to investigate root cause before continuing Stage 3 finalization.

**Eliminate the padding-layout shortcut introduced by `PadTo256` padding (all -1 region after normalization).**

Status:
- `benign_images_nataraj_v3` is present on Ubuntu at `/home/alucard-00/EC499/benign_images_nataraj_v3/` (18,537 PNG files).
- v3 benign is integrated into `Project_Resourse/archive/malimg_dataset/*/benign/`.
- Dataset split sizes (CURRENT):
  - Train: 22,288 (14,829 benign + 7,459 malware)
  - Val: 2,663 (1,853 benign + 810 malware)
  - Test: 2,702 (1,855 benign + 847 malware)

---

## 3. Latest Numerical Results (Biased Dataset — for comparison only)

### Stage 2 Results

| Metric | Value |
|---|---|
| Training samples | 17,976 (10,517 benign + 7,459 malware) |
| Best val loss epoch | Epoch 2 |
| Best val loss | 0.0561 |
| Clean test accuracy | 99.08% |
| Precision | 99.05% |
| Recall | 98.58% |
| F1 | 98.82% |
| FPR | 0.61% |

### Stage 3 Attack Evaluation (Clean ResNet-18)

| Attack | Epsilon | Accuracy |
|---|---|---|
| Clean | — | 99.08% |
| FGSM | 0.01 | 77.95% |
| FGSM | 0.05 | 20.20% |
| FGSM | 0.10 | 13.82% |
| PGD (10 steps) | 0.01 | 68.65% |
| PGD (40 steps) | 0.05 | 8.37% |

### Stage 3 Adversarial Training (ResNet-18)

| Epoch | Train Loss | Train Acc | Val Clean | Val Robust |
|---|---|---|---|---|
| 1 | 0.3447 | 89.67% | 97.98% | 74.58% |
| 2 | 0.1508 | 96.00% | 98.63% | 95.76% |
| 3 | 0.1111 | 97.10% | 98.82% | 96.23% |
| 4 | 0.0953 | 97.63% | 99.01% | **97.60%** ⭐ |
| 5 | 0.0856 | 97.74% | 99.01% | 97.55% |

### Split-Class Evasion Rate

| Model | Clean Recall | Under PGD ε=0.05 (40-step) | Evasion Rate |
|---|---|---|---|
| Clean Baseline | 98.58% (835/847) | 2.48% (21/847) | 97.52% |
| AT Model | 98.47% (834/847) | 95.51% (809/847) | **4.49%** |

### Black-Box Transfer Test

| Attack Type | AT Model Accuracy |
|---|---|
| White-box PGD | 97.41% |
| Black-box transfer | 99.12% |
| Gap | 1.71 pp |

**Interpretation:** Gap < 5 pp → gradient masking ruled out → robustness is genuine.

### Logit Distribution (Bias Indicator)

| Class | Mean Logit | Std | Median |
|---|---|---|---|
| Benign | -11.96 | 4.10 | -12.20 |
| Malware | +9.90 | 3.97 | +11.04 |
| **Gap** | **21.86 units** | | |

**Healthy range: 5–10 units.** Gap of 22 units confirms source bias.

### Accuracy vs Epsilon (Summary)

| Epsilon | Clean Model | AT Model |
|---|---|---|
| 0.000 | 99.08% | 99.17% |
| 0.010 | 67.22% | 98.98% |
| 0.050 | 9.02% | 97.46% |
| 0.100 | 3.56% | 56.91% |
| 0.200 | 1.39% | 12.07% |

---

## 4. Known Problems

### Problem 1 — Padding-Layout Shortcut from PadTo256 (PRIMARY ISSUE)
**Status:** CONFIRMED, mitigation in progress
**Evidence:** Padding-only logistic regression using only padding-derived features achieves high AUC on v3-integrated dataset.
**Impact:** Trained ResNet-18 shows a large logit gap mean (~22 units) despite high test accuracy, consistent with shortcut-driven separation.
**Mitigations being tested:**
- `train.py` optional `PAD_NEUTRALIZE_SAMPLER=1` (WeightedRandomSampler based on padding-only LR propensity)
- `dataset_loader.py` optional `PAD_NOISE=1` (randomize only padded bottom-right region)
- Pad-matched dataset subset (`archive/malimg_dataset_padmatched_v2`) where padding-only AUC ~0.52 (small-scale control)

### Problem 2 — .git Folder Size (40 GB local)
**Status:** PENDING CLEANUP
**Cause:** Large files temporarily staged before .gitignore was finalized. Removed from staging but git retained internal copies.
**Fix:**
```bash
cd /home/alucard-00/EC499
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

### Problem 3 — config.py Missing New Constants
**Status:** PARTIALLY FIXED on Ubuntu, may not be committed to GitHub
**Fix:** Ensure config.py contains:
```python
RESNET_ADV_TRAINED_MODEL_PATH_STR = str(MODEL_OUTPUT_DIR / "resnet18_adversarially_trained.pth")
EFFICIENTNET_CLEAN_MODEL_PATH_STR = str(MODEL_OUTPUT_DIR / "efficientnet_b0_clean_vulnerable.pth")
EFFICIENTNET_ADV_TRAINED_MODEL_PATH_STR = str(MODEL_OUTPUT_DIR / "efficientnet_b0_adversarially_trained.pth")
```

---

## 5. Immediate Next Steps (In Order)

### Step 1 — Quantify Shortcut Strength (v3 dataset)
Run padding-only shortcut diagnostic on a representative subset (fast) and optionally full dataset (slow):
`./Project_Resourse/venv/bin/python Project_Resourse/verify_padding_shortcut.py --max-train 8000 --max-val 2000 --max-test 2000 --no-plots`

### Step 2 — Run Full Diagnostics on Current Model
`./Project_Resourse/venv/bin/python Project_Resourse/tmp_split_diag_full.py`

Record:
- test accuracy
- ROC-AUC
- logit gap mean

### Step 3 — Run Mitigation Experiments (one at a time)
1) Pad-matched subset training:
`DATA_DIR_OVERRIDE=/home/alucard-00/EC499/Project_Resourse/archive/malimg_dataset_padmatched_v2 ./Project_Resourse/venv/bin/python Project_Resourse/train.py`

2) Full dataset with sampling neutralization:
`PAD_NEUTRALIZE_SAMPLER=1 ./Project_Resourse/venv/bin/python Project_Resourse/train.py`

3) Full dataset with padding noise transform:
`PAD_NOISE=1 PAD_NOISE_MODE=uniform ./Project_Resourse/venv/bin/python Project_Resourse/train.py`

### Step 4 — Decide Pass/Fail Criterion
Primary criterion for "shortcut removed": padding-only AUC near 0.5 AND logit gap mean reduced (target 5-10 units) without catastrophic accuracy drop.

### Step 7 — Adversarial Training
```bash
/home/alucard-00/EC499/venv/bin/python Project_Resourse/adversarial_train.py
/home/alucard-00/EC499/venv/bin/python Project_Resourse/adversarial_train_efficientnet.py
```

### Step 8 — Regenerate Fixed Adversarial Test Set
```bash
/home/alucard-00/EC499/venv/bin/python Project_Resourse/generate_adversarial_test_set.py
```

### Step 9 — Final Comparison Table
Report clean vs AT for both architectures. Target evasion rate: 15–35%.

---

## 6. Models on Disk

| File | Architecture | Dataset | Status |
|---|---|---|---|
| `resnet18_clean_vulnerable.pth` | ResNet-18 | Biased benign | Valid, 11.18M params |
| `resnet18_adversarially_trained.pth` | ResNet-18 | Biased benign | Valid, 11.18M params |
| `efficientnet_b0_clean_vulnerable.pth` | EfficientNet-B0 | Biased benign | Valid, 4.01M params |
| `custom_cnn_clean_vulnerable.pth` | CustomCNN | Biased benign | Legacy, keep for reference |

CURRENT run logs (local): `run_logs/train_resnet18_full_padneutralized_trainpy.log`, `run_logs/full_model_diag_after_padneutralize.log`

All models: conv1.weight shape [64,1,7,7] for ResNet or [32,1,3,3] for EfficientNet — confirms grayscale adaptation correct.

---

## 7. MaleX Track Status (2026-03-30)

### Migration status

- MaleX integration prompt execution reached Stage 3.
- MaleX split builder and validation scripts are in place.
- Active dataset: `Project_Resourse/archive/malex_dataset/`.

### Stage 2 (MaleX clean baseline) — completed

- Script: `Project_Resourse/train.py`
- Log: `run_logs/train_resnet18_malex_stage2.log`
- Model: `Project_Resourse/models/resnet18_malex_clean_vulnerable.pth`
- Early stopping: epoch 7 (best epoch 2)
- Best val loss: 0.3695
- Final epoch metrics: Train 96.91% / Val 83.16%

### Stage 3 Part 1 (attack vulnerability) — completed

- Script: `Project_Resourse/evaluate_attacks.py`
- Log: `run_logs/evaluate_attacks_malex_stage3.log`
- Results file: `Project_Resourse/logs/attack_evaluation_results.txt`
- Clean accuracy: 82.26%
- FGSM and PGD robustness collapses to near-zero at moderate epsilon.

### Stage 3 Part 2 (adversarial training) — paused

- Decision: paused until base-model issue is diagnosed.
- Reason: candidate base models are not improving sufficiently for robust final benchmark.

### Base-model experiment checkpoint summary (new)

- 3C2D run log: `run_logs/train_3c2d_malex_fixed.log`
  - Best val loss: 0.3246 at epoch 49/50
  - Best val acc: 85.62% at epoch 52/70
  - End state: early stopped at epoch 60/70 (train acc 88.40%, val acc 85.50%)

- Pretrained ResNet-18 run log: `run_logs/train_resnet_pretrained_malex.log`
  - Best val loss (current): 0.3354 at epoch 3/50
  - Best val acc (current): 85.53% at epoch 4/50
  - Latest before stop: epoch 7/50 (train acc 95.59%, val acc 85.43%, val loss 0.4565)

Conclusion:
- 3C2D: plateau after epoch 49 region.
- Pretrained ResNet-18: overfitting trend.
- Next action: root-cause analysis before re-running Task C/D final pipeline.

### Important note on split leakage check

- `check_malex_hash_overlaps.py` currently reports cross-split hash overlaps.
- Training was continued by user request for momentum, but split dedup remediation remains an open methodological task.

### Prompt-aligned phase report (MaleX integration)

The following captures the exact phased flow requested in `AGENT_PROMPT_MaleX_Integration.md`.

| Phase | Status | Key output |
|---|---|---|
| Phase 1 — Verify source images | PASS | Benign and malware sources validated at 256x256, no corruption |
| Phase 2 — Build balanced split | PASS | Built `archive/malex_dataset` with balanced 80/10/10 class-preserving split |
| Phase 3 — Update config.py | PASS | MaleX dataset/model path constants added and import check passed |
| Phase 4 — Update dataset_loader.py | PASS | MaleX loader self-test passed: shape/range/label assertions |
| Phase 5 — Update train/evaluate/adversarial scripts | PASS | All three imports passed after MaleX rewiring |
| Phase 6A — Hash overlap check | FAIL | Cross-split overlaps detected in benign and malware hashes |
| Phase 6B — Label/tensor check | PASS | Labels and tensor assertions passed |
| Phase 6C — Shuffle sanity check | PASS | Final shuffled-label validation stayed near chance (50.0%) |
| Phase 7 — Stage 2 clean training | PASS | Completed with early stopping at epoch 7 |
| Stage 3 Part 1 — Attack evaluation | PASS | Clean 82.26%, severe FGSM/PGD collapse confirms vulnerability |
| Stage 3 Part 2 — Adversarial training | RUNNING | Active log at `run_logs/adversarial_train_malex_stage3.log` |
