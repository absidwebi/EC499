# EC499 - Adversarial Robustness in Deep Learning-Based Malware Detection

Student: Abdulsalam Ashraf Aldwebi (ID: 2210245306)
Supervisor: Dr. Suad Elgeder | University of Tripoli
GitHub: https://github.com/absidwebi/EC499
Primary machine: Ubuntu Linux, RTX 4060, /home/alucard-00/EC499
Primary interpreter: /home/alucard-00/EC499/venv/bin/python

Last updated: 2026-04-11 (FGSM continuation finalized + Stage 2/Stage 3 metric pack refreshed)

---

## 1. Project Objective

Build a reproducible malware-image binary classifier and rigorously evaluate adversarial vulnerability and defense for thesis-grade reporting.

Binary task:
- Benign = 0
- Malware = 1

Hard constraints enforced in active code:
- Single-logit output with BCEWithLogitsLoss
- Adversarial clamp range [-1.0, 1.0]
- No geometric augmentation in byteplot pipeline
- num_workers = 0 on Linux/CUDA runs
- Paths imported from config.py (no hardcoded active script paths)

---

## 2. Active Data Track (MaleX Byteplot)

Source roots:
- Benign: /home/alucard-00/EC499/ben_byteplot_imgs_zipped/byteplot_imgs_RxR/256/
- Malware: /home/alucard-00/EC499/mal_byteplot_imgs_zipped/byteplot_imgs_RxR/256/

Working split root:
- /home/alucard-00/EC499/Project_Resourse/archive/malex_dataset/

Verified split sizes:
- Train: 287,560 (143,780 benign + 143,780 malware)
- Val: 33,015
- Test: 33,004 (17,153 benign + 15,851 malware)

Primary fixed adversarial set:
- Project_Resourse/adversarial_test_set_malex/fgsm_eps0.05: 15,851 PNG
- Project_Resourse/adversarial_test_set_malex/pgd_eps0.05_steps40: 15,851 PNG

Legacy fixed set (kept for reference only):
- Project_Resourse/adversarial_test_set/fgsm_eps0.05: 847 PNG
- Project_Resourse/adversarial_test_set/pgd_eps0.05_steps40: 847 PNG

---

## 3. Model Lineup and Verified Parameter Counts

Canonical model set for final Stage 3 comparison:
- 3C2D clean baseline
- 3C2D adversarially trained (PGD)
- 3C2D adversarially trained (FGSM)

Verified parameter counts (run on local environment, 2026-04-11):
- MaleX3C2D: 1,273,345
- ResNet-18 pretrained grayscale: 11,170,753

Artifacts:
- Project_Resourse/models/3c2d_malex_clean_vulnerable.pth
- Project_Resourse/models/3c2d_malex_adversarially_trained.pth
- Project_Resourse/models/3c2d_malex_fgsm_adversarially_trained.pth
- Project_Resourse/models/at_3c2d_full_checkpoint.pth
- Project_Resourse/models/at_3c2d_fgsm_full_checkpoint.pth

---

## 4. Stage 2 Baseline Metrics (Clean 3C2D)

Latest dedicated Stage 2 clean-only artifact:
- run_logs/stage2_eval_3c2d_clean_only_20260411_002438.log
- Project_Resourse/logs/stage2_eval_3c2d_clean_only_20260411_002438.txt

Results:
- Accuracy: 85.29%
- Confusion: TN=14947 FP=2206 FN=2648 TP=13203
- Precision (benign/malware): 0.8495 / 0.8568
- Recall (benign/malware): 0.8714 / 0.8329
- Balanced accuracy: 0.8522
- MCC: 0.7053
- F1 macro / malware: 0.8525 / 0.8447
- AUC-ROC: 0.9316
- TPR@FPR<=1% / <=5%: 0.5138 / 0.7018

---

## 5. Stage 3 Defense Track (Canonical 2026-04-11)

### 5.1 PGD adversarial training

Status: complete.

Canonical training summary:
- Per-epoch log: Project_Resourse/logs/adversarial_training_log_3c2d.txt
- Full checkpoint: Project_Resourse/models/at_3c2d_full_checkpoint.pth
- Best robust val acc: 74.123883% at epoch 35
- Early stop point: epoch 40/50

### 5.2 FGSM adversarial training

Status: complete with continuation.

Continuation details:
- Canonical run log: run_logs/adversarial_train_fgsm_20260401_052733.log
- Original segment: epochs 1-20, best 72.7306% at epoch 19
- Continuation segment: resumed from epoch 20 and continued to epoch 32
- Continuation mode used best epoch-19 weights as restart state
- Early stopping triggered at epoch 32 (with /80 max epoch budget)
- New best robust val acc: 74.329850% at epoch 27

Canonical FGSM checkpoint decision:
- Canonical FGSM best checkpoint is now the continued model at:
  Project_Resourse/models/3c2d_malex_fgsm_adversarially_trained.pth
- Canonical full checkpoint state is:
  Project_Resourse/models/at_3c2d_fgsm_full_checkpoint.pth
  with epoch_zero_based=31, best_epoch=27, best_robust_val_acc=74.329850.

Discrepancy resolution note (77.33 vs 77.96 clean FGSM accuracy):
- 77.33% appears in older fixed-set evaluation artifacts before continuation.
- 77.9633% appears in post-continuation artifacts and is now canonical.

### 5.3 Stage 3 fixed-set all-model comparison (latest)

Evaluator:
- Project_Resourse/evaluate_attacks_fixed.py --model all

Latest artifacts:
- run_logs/stage3_fixed_eval_all_three_models_20260411_001931.log
- Project_Resourse/logs/fixed_adv_eval_all_three_models_20260411_001931.txt
- Project_Resourse/logs/stage3_complete_comparison_all3_20260411_001931.txt
- Project_Resourse/logs/fixed_adv_eval_all.txt

Key comparison metrics:

| Model | Clean Accuracy | FGSM Recall | FGSM Evasion | PGD Recall | PGD Evasion |
|---|---:|---:|---:|---:|---:|
| 3C2D Clean | 85.2927% | 15.29% | 84.71% | 0.53% | 99.47% |
| 3C2D AT (PGD) | 80.0297% | 74.97% | 25.03% | 74.94% | 25.06% |
| 3C2D AT (FGSM) | 77.9633% | 68.92% | 31.08% | 68.81% | 31.19% |

Interpretation:
- PGD AT remains strongest on fixed-set robustness.
- FGSM AT improved materially after continuation and is now stronger than its pre-continuation state.
- Clean model retains highest clean accuracy but collapses under fixed adversarial attacks.

---

## 6. Log and Curve Completeness Artifacts

Created for reporting completeness from run_logs/txt reconstruction:
- Project_Resourse/logs/3C2D_MaleX_clean_Baseline.txt
- Project_Resourse/logs/Resnet18_MaleX_clean_Baseline.txt
- Project_Resourse/logs/3C2D MaleX clean Baseline.txt
- Project_Resourse/logs/Resnet18 MaleX clean Baseline.txt
- Project_Resourse/logs/adversarial_training_log_3c2d_reconstructed.txt
- Project_Resourse/logs/adversarial_training_log_fgsm_reconstructed.txt
- Project_Resourse/logs/training_curve_3c2d_clean_baseline.png
- Project_Resourse/logs/training_curve_3c2d_fixed.png
- Project_Resourse/logs/training_curve_resnet18_malex_clean_baseline.png
- Project_Resourse/logs/adversarial_training_curve_3c2d.png (reconstructed full-history plot)
- Project_Resourse/logs/adversarial_training_curve_fgsm.png (reconstructed full-history plot)

---

## 7. Code Changes Since Last Documentation Sync

Updated Python script in this cycle:
- Project_Resourse/adversarial_train_fgsm.py

What changed:
- Added FGSM_NUM_EPOCHS env override.
- Added FGSM_EARLY_STOP_PATIENCE env override.
- Added FGSM_RESUME_FROM_BEST_WEIGHTS mode.
- Added logic to restore best robust weights for continuation while preserving checkpoint epoch/log continuity.

Why:
- Needed controlled continuation from epoch 20 using best epoch-19 weights and append-only logging to canonical FGSM run log.

No new model architecture changes were introduced in this cycle.

---

## 8. Stage 4 Inference Status

Stage 4 remains implemented and previously validated.

Core files:
- Project_Resourse/inference.py
- Project_Resourse/app.py
- Project_Resourse/templates/index.html
- Project_Resourse/Dockerfile

Validated behavior:
- /health returns status JSON
- /compare rejects invalid non-PE inputs with HTTP 400
- /compare accepts valid PE and returns comparison payload (clean/adv visuals and model outputs)

Environment note:
- Bridge networking with -p is required for browser access in this environment.

---

## 9. Open Risks and Unchecked Requirements

1) Cross-split overlap remediation is still not closed.
2) Stage 4 committee/demo bundle packaging is still pending.
3) evaluate_base_models_testset.py has a relative output-path assumption when run from inside Project_Resourse; direct script execution from that cwd can fail JSON write.
4) Sidecar metadata is implemented in PGD trainer but is not uniformly standardized across all historical artifacts.
5) Workspace still contains large unrelated untracked binary/data files that must stay out of scoped commits.

---

## 10. Immediate Next Actions

1) Lock report tables to canonical 2026-04-11 artifacts listed above.
2) Re-run any remaining final audit scripts only against canonical checkpoints.
3) Resolve cross-split overlap risk and refresh diagnostics once complete.
4) Assemble Stage 4 appendix package (sample files, exact commands, expected output captures).
