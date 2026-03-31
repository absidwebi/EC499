# CURRENT_STATE.md
# EC499 - Current Project State

Last updated: 2026-03-31

---

## 1. Stage Progress Snapshot

| Stage | Status | Notes |
|---|---|---|
| Stage 1 - Dataset Preparation | DONE | MaleX split pipeline active in archive/malex_dataset |
| Stage 2 - Base Model Selection | DONE (this update) | Final test-set comparison completed between 3C2D and pretrained ResNet-18 |
| Stage 3 Part 1 - Attack Evaluation | DONE | FGSM/PGD grid executed on selected clean baseline |
| Stage 3 Part 2 - Adversarial Training | RUNNING | Active 3C2D adversarial training process detected |
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

## 5. Live Runtime State

Observed at update time:
- Active process: `Project_Resourse/venv/bin/python Project_Resourse/adversarial_train.py`
- PID: 328551

This indicates Stage 3 Part 2 training is currently in progress.

---

## 6. Known Open Risks

1) Cross-split overlap issue in MaleX remains unresolved and must be fixed before final thesis-grade robustness claims.
2) Final robust metrics are pending completion of current adversarial training run.

---

## 7. Immediate Next Actions

1) Finish current adversarial training run and capture best robust validation checkpoint/metrics.
2) Evaluate robust checkpoint with same FGSM/PGD grid for direct clean-vs-robust comparison.
3) Publish final Stage 3 summary table and interpretation.
4) Resolve/verify split-overlap remediation and re-run integrity checks.
