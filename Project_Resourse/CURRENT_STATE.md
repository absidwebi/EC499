# CURRENT_STATE.md
# EC499 - Current Project State

Last updated: 2026-04-01

---

## 1. Stage Progress Snapshot

| Stage | Status | Notes |
|---|---|---|
| Stage 1 - Dataset Preparation | DONE | MaleX split pipeline active in archive/malex_dataset |
| Stage 2 - Base Model Selection | DONE (this update) | Final test-set comparison completed between 3C2D and pretrained ResNet-18 |
| Stage 3 Part 1 - Attack Evaluation | DONE | FGSM/PGD grid executed on selected clean baseline |
| Stage 3 Part 2 - Adversarial Training | DONE (baseline defense run) | Completed 3C2D defended run with saved robust checkpoint |
| Stage 4 - Documentation + Git sync | IN PROGRESS | Context docs and commit/push in this cycle |

---

## 2. What Changed Since 2026-03-30

### 2.1 New/updated scripts

1) New: `Project_Resourse/evaluate_base_models_testset.py`
- Added comprehensive test-set evaluator for:
  - MaleX3C2D
  - ResNet-18 pretrained grayscale
- Metrics: accuracy, confusion matrix, F1 macro, F1 malware, ROC-AUC, logit mean/std by class.

2) Updated: `Project_Resourse/evaluate_attacks.py`
- Added model selection switch (`3c2d`, `resnet`, `resnet_pretrained`).
- Default switched to `3c2d`.
- Output file now model-tagged: `attack_evaluation_results_3c2d.txt`.

3) Updated: `Project_Resourse/adversarial_train.py`
- Added same model selection switch and checkpoint bundle mapping.
- Default switched to `3c2d`.
- Added `torch.manual_seed(42)` for deterministic startup.
- Fixed log file variable bug (runtime `log_path` now used correctly).

### 2.2 New run artifact

- `Project_Resourse/base_model_testset_results.json` generated successfully.
- `Project_Resourse/logs/attack_comparison_3c2d_before_after_stage3.json` generated from clean-vs-defended evaluation.

---

## 3. Latest Measured Results (Authoritative)

Source file:
- `Project_Resourse/base_model_testset_results.json`

| Model | Accuracy | F1 Macro | F1 Malware | AUC-ROC |
|---|---:|---:|---:|---:|
| MaleX3C2D | 85.2927% | 0.8525 | 0.8447 | 0.9316 |
| ResNet-18 Pretrained | 85.2594% | 0.8520 | 0.8430 | 0.9326 |

Model recommendation chosen for Stage 3:
- MaleX3C2D

Reason:
- Slight edge on accuracy and malware-class F1 while overall quality remains nearly tied.

Benchmark comparison used in decision:
- Paper 3C2D benchmark: 83.60% -> observed 85.29% (+1.69pp)
- Paper ResNet-18 benchmark: 82.52% -> observed 85.26% (+2.74pp)

---

## 4. Stage 3 Attack Results (Selected Baseline)

Source:
- `Project_Resourse/logs/attack_evaluation_results_3c2d.txt`

| Attack | Config | Accuracy |
|---|---|---:|
| Clean | none | 85.29% |
| FGSM | eps=0.01 | 56.93% |
| FGSM | eps=0.02 | 34.94% |
| FGSM | eps=0.05 | 10.45% |
| FGSM | eps=0.10 | 3.49% |
| PGD | eps=0.01, 10 steps | 46.62% |
| PGD | eps=0.02, 20 steps | 13.32% |
| PGD | eps=0.05, 40 steps | 0.61% |

Interpretation:
- Vulnerability confirmed; baseline is appropriate input for adversarial training.

---

## 5. Stage 3 Defense Results (Completed Run)

Training run log:
- `run_logs/adversarial_train_ 3C2D_Fixed_malex_stage3.log`

Checkpoint saved:
- `Project_Resourse/models/3c2d_malex_adversarially_trained.pth`

Training summary:
- Final epoch: 5/5
- Train Loss: 0.5619
- Train Acc: 65.93%
- Val Clean: 73.29%
- Val Robust: 64.91%
- Best Robust Val Accuracy: 64.91%

---

## 6. Before vs After (Clean vs Defended)

Evaluation run log:
- `run_logs/evaluate_attacks_3c2d_clean_vs_defended_stage3.log`

Comparison artifact:
- `Project_Resourse/logs/attack_comparison_3c2d_before_after_stage3.json`

| Attack | Config | Before | After | Delta |
|---|---|---:|---:|---:|
| Clean | - | 85.29% | 73.07% | -12.23pp |
| FGSM | 0.01 | 56.93% | 71.63% | +14.70pp |
| FGSM | 0.02 | 34.94% | 70.18% | +35.24pp |
| FGSM | 0.05 | 10.45% | 66.26% | +55.81pp |
| FGSM | 0.10 | 3.49% | 61.16% | +57.67pp |
| PGD | 0.01, 10 steps | 46.64% | 71.55% | +24.92pp |
| PGD | 0.02, 20 steps | 13.32% | 69.92% | +56.60pp |
| PGD | 0.05, 40 steps | 0.64% | 64.61% | +63.98pp |

Interpretation:
- Adversarial robustness improved strongly for FGSM and PGD.
- Clean accuracy dropped, consistent with robustness tradeoff.

---

## 7. Code Delta Since Previous State Update

Updated script:
- `Project_Resourse/adversarial_train.py`

New changes in this delta:
- Added full resume-capable checkpoint support:
  - save_full_checkpoint
  - load_full_checkpoint
- Added training-curve output function:
  - plot_training_curves
- Reworked training loop to support:
  - resume from saved epoch
  - per-epoch full checkpoint save
  - per-epoch log persistence
  - robust-validation early stopping (`EARLY_STOP_PATIENCE=5`)
- Increased configured training budget from 5 to 20 epochs for future runs.

---

## 8. Known Open Risks

1) Cross-split overlap issue in MaleX remains unresolved and must be fixed before final thesis-grade robustness claims.
2) Latest robust validation is 64.91%, slightly below a 65% target threshold.

---

## 9. Immediate Next Actions

1) Continue/extend adversarial training with the new resume-capable script to push robust validation above 65% if required.
2) Re-run before/after FGSM/PGD comparison after any extended run.
3) Resolve/verify split-overlap remediation and re-run integrity checks.
