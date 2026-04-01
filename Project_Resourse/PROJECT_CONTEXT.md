# PROJECT_CONTEXT.md
# EC499 - Adversarial Robustness in Deep Learning-Based Malware Detection

Student: Abdulsalam Ashraf Aldwebi (ID: 2210245306)
Supervisor: Dr. Suad Elgeder | University of Tripoli
GitHub: https://github.com/absidwebi/EC499
Primary Machine: Ubuntu, RTX 4060, /home/alucard-00/EC499/
Python Environment (active and verified): /home/alucard-00/EC499/Project_Resourse/venv/

Last updated: 2026-04-01

---

## 1. Project Objective

Build a reproducible malware image classification pipeline and evaluate adversarial vulnerability/defense under controlled methodology.

Binary task:
- Benign = 0
- Malware = 1

Core rule set maintained in all active scripts:
- Single-logit output + BCEWithLogitsLoss
- Adversarial clamp range is always [-1.0, 1.0]
- No interpolation artifacts in malware-image preprocessing logic
- `num_workers=0` for Linux CUDA stability

---

## 2. Active Data Track

Current active branch for Stage 2/3 is MaleX byteplot data.

Source roots:
- Benign: /home/alucard-00/EC499/ben_byteplot_imgs_zipped/byteplot_imgs_RxR/256/
- Malware: /home/alucard-00/EC499/mal_byteplot_imgs_zipped/byteplot_imgs_RxR/256/

Working split root:
- /home/alucard-00/EC499/Project_Resourse/archive/malex_dataset/

MaleX split sizes:
- Train: 287,560 (143,780 benign + 143,780 malware)
- Val: 35,944 (17,972 benign + 17,972 malware)
- Test: 35,946 (17,973 benign + 17,973 malware)

Note:
- Hash-overlap remediation across splits remains a tracked methodological risk item.

---

## 3. Model Candidates Used in Stage 2 Final Selection

Evaluated candidates for final clean baseline selection before Stage 3 adversarial training:
- MaleX3C2D (fixed, resume-capable run lineage)
- ResNet-18 pretrained grayscale variant

Checkpoints used:
- 3C2D clean: Project_Resourse/models/3c2d_malex_clean_vulnerable.pth
- ResNet-18 pretrained clean: Project_Resourse/models/resnet18_malex_pretrained_clean.pth

---

## 4. Stage 2 Final Evaluation (Completed 2026-03-31)

Evaluation script created and executed:
- Project_Resourse/evaluate_base_models_testset.py

Produced artifact:
- Project_Resourse/base_model_testset_results.json

### 4.1 Test-set metrics (exact)

| Model | Accuracy | F1 Macro | F1 Malware | AUC-ROC | Benign Logit Mean +- Std | Malware Logit Mean +- Std |
|---|---:|---:|---:|---:|---:|---:|
| MaleX3C2D | 85.2927% | 0.8525 | 0.8447 | 0.9316 | -2.6726 +- 2.5572 | 3.7370 +- 4.1915 |
| ResNet-18 Pretrained | 85.2594% | 0.8520 | 0.8430 | 0.9326 | -2.6028 +- 2.3394 | 3.7453 +- 4.0506 |

Confusion matrices:
- 3C2D: [[14947, 2206], [2648, 13203]]
- ResNet-18 pretrained: [[15075, 2078], [2787, 13064]]

### 4.2 Benchmark comparison (paper table targets)

Paper baselines provided for comparison:
- 3C2D benchmark: 83.60%
- ResNet-18 benchmark: 82.52%

Observed gains on this run:
- 3C2D: +1.69 percentage points over 83.60%
- ResNet-18 pretrained: +2.74 percentage points over 82.52%

Selection decision for Stage 3:
- Selected: MaleX3C2D
- Rationale: Slight edge in accuracy and malware-focused F1, with essentially tied macro quality.

---

## 5. Stage 3 Preparation and Execution Status

### 5.1 Stage 3 Part 1 attack evaluation (completed)

Script:
- Project_Resourse/evaluate_attacks.py

Log artifact:
- Project_Resourse/logs/attack_evaluation_results_3c2d.txt

Latest results (3C2D clean baseline):
- Clean: 85.29%
- FGSM eps 0.01: 56.93%
- FGSM eps 0.02: 34.94%
- FGSM eps 0.05: 10.45%
- FGSM eps 0.10: 3.49%
- PGD eps 0.01 (10 steps): 46.62%
- PGD eps 0.02 (20 steps): 13.32%
- PGD eps 0.05 (40 steps): 0.61%

Interpretation:
- Clean baseline remains strongly vulnerable under stronger FGSM/PGD as expected.

### 5.2 Stage 3 Part 2 adversarial training (completed)

Script:
- Project_Resourse/adversarial_train.py

Run log used for this completed run:
- run_logs/adversarial_train_ 3C2D_Fixed_malex_stage3.log

Final epoch summary from run log:
- Epoch 5/5
- Train Loss: 0.5619
- Train Acc: 65.93%
- Val Clean: 73.29%
- Val Robust: 64.91%

Best robust validation metric:
- Best Robust Val Accuracy: 64.91%

Saved robust checkpoint:
- Project_Resourse/models/3c2d_malex_adversarially_trained.pth

Training summary log:
- Project_Resourse/logs/adversarial_training_log_3c2d.txt

### 5.3 Stage 3 clean-vs-defended attack comparison (completed)

Evaluation run log:
- run_logs/evaluate_attacks_3c2d_clean_vs_defended_stage3.log

Comparison artifact:
- Project_Resourse/logs/attack_comparison_3c2d_before_after_stage3.json

Before (clean baseline) vs After (defended model):

| Attack | Config | Before | After | Delta |
|---|---|---:|---:|---:|
| Clean | - | 85.29% | 73.07% | -12.23pp |
| FGSM | 0.01 | 56.93% | 71.63% | +14.70pp |
| FGSM | 0.02 | 34.94% | 70.18% | +35.24pp |
| FGSM | 0.05 | 10.45% | 66.26% | +55.81pp |
| FGSM | 0.10 | 3.49% | 61.16% | +57.67pp |
| PGD | 0.01 (10 steps) | 46.64% | 71.55% | +24.92pp |
| PGD | 0.02 (20 steps) | 13.32% | 69.92% | +56.60pp |
| PGD | 0.05 (40 steps) | 0.64% | 64.61% | +63.98pp |

Interpretation:
- Robustness under FGSM/PGD improved substantially across all attack strengths.
- Clean accuracy decreased after adversarial training, reflecting robustness-accuracy tradeoff.

---

## 6. Exact Code Changes Since Previous Context Update

### 6.1 New script

1) Project_Resourse/evaluate_base_models_testset.py
- Added to perform comprehensive test-set evaluation for the two selected base models.
- Computes:
  - Accuracy
  - Confusion matrix
  - F1 macro
  - F1 malware class
  - ROC-AUC
  - Per-class logit mean/std
- Saves structured output JSON to Project_Resourse/base_model_testset_results.json.

Why:
- Required to finalize Stage 2 model selection with reproducible metrics and direct benchmark comparison.

### 6.2 Updated script

2) Project_Resourse/evaluate_attacks.py
- Added model variant routing:
  - `3c2d`
  - `resnet`
  - `resnet_pretrained`
- Default variant set to `3c2d` for current Stage 3 pipeline.
- Added model-tagged output naming:
  - attack_evaluation_results_3c2d.txt

Why:
- Prevent hardcoded model mismatch and make attack evaluation explicitly aligned with selected Stage 2 winner.

3) Project_Resourse/adversarial_train.py
- Previous update (already integrated):
  - Added model variant routing and model-tagged logging.
  - Added clean/robust checkpoint bundle mapping and reproducibility seed.
- New hardening update (2026-04-01):
  - Increased `NUM_EPOCHS` from 5 to 20.
  - Added `EARLY_STOP_PATIENCE=5` on robust validation accuracy.
  - Added full resume capability via checkpoint state:
    - epoch index
    - model weights
    - optimizer state
    - best robust metric and epoch
    - no-improve counter
    - cumulative log lines
  - Added `RESUME_IF_CHECKPOINT_EXISTS` control flag.
  - Added per-epoch full checkpoint save path:
    - `Project_Resourse/models/at_3c2d_full_checkpoint.pth` (variant-tagged path construction)
  - Added per-epoch training curve generation:
    - `Project_Resourse/logs/adversarial_training_curve_3c2d.png` (generated on next run with updated script)
  - Added per-epoch persistent log writing for safer interruption recovery.

Why:
- Ensure Stage 3 defense training uses intended baseline model and saves outputs under unambiguous names.

---

## 7. Known Open Items (Current)

1) MaleX split overlap risk:
- Cross-split overlap finding remains open and must be resolved before final thesis-grade robustness claims.

2) Stage 3 robust run quality threshold:
- Latest completed defended run reached 64.91% robust validation accuracy, slightly below a 65% target threshold.
- Updated resume-capable script is now ready for longer continuation without restarting from epoch 1.

3) Large untracked binaries in workspace root:
- Present locally (dataset extractions/archives) and intentionally excluded from commit scope.

---

## 8. Immediate Next Actions (Operational)

1) Run extended adversarial training using updated resume-capable `adversarial_train.py` (20-epoch budget + early stop).
2) Re-run clean-vs-defended comparison after extended training and track whether robust val >= 65% is achieved.
3) Resolve remaining split-overlap methodological issue before thesis-final benchmark lock.
4) Keep live run monitoring standardized via run_logs tee pipeline for all long training jobs.
