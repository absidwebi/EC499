# EC499 Folder Structure Overview

Last updated: 2026-04-11 (aligned with finalized 3-model Stage 3 comparison and FGSM continuation)

This document maps the current workspace layout and highlights canonical files for reporting and reproducibility.

---

## 1. Top-Level Layout

EC499/
- AGENTS.md
- EC499_Folder_Structure.md
- Project_Resourse/
- run_logs/
- ben_byteplot_imgs_zipped/
- mal_byteplot_imgs_zipped/
- benign_images_256x256/
- benign_images_nataraj_v3/
- benign_images_test/
- train_list.txt, val_list.txt
- train_dirs.txt, val_dirs.txt

Commit hygiene:
- Large archives, datasets, binaries, and generated PNG corpora should not be committed in normal scoped commits.

---

## 2. Active Project_Resourse Code Files

Core pipeline:
- config.py
- dataset_loader.py
- models.py
- train.py
- train_efficientnet.py
- adversarial_train.py
- adversarial_train_fgsm.py
- adversarial_train_efficientnet.py

Evaluation:
- evaluate_base_models_testset.py
- evaluate_attacks.py
- evaluate_attacks_fixed.py
- generate_adversarial_test_set.py

Stage 4 service:
- inference.py
- app.py
- templates/index.html
- Dockerfile

---

## 3. Key Subdirectories Under Project_Resourse

- archive/
  - malex_dataset/ (active split root)
- logs/
  - canonical training logs, evaluation txt outputs, and curve PNGs
- models/
  - clean and adversarially trained checkpoints
- adversarial_test_set_malex/
  - primary fixed adversarial test subsets for deterministic Stage 3 comparison
- adversarial_test_set/
  - legacy smaller fixed adversarial subsets
- templates/
  - Stage 4 UI template files

---

## 4. Canonical Model Artifacts

- Project_Resourse/models/3c2d_malex_clean_vulnerable.pth
- Project_Resourse/models/3c2d_malex_adversarially_trained.pth
- Project_Resourse/models/3c2d_malex_fgsm_adversarially_trained.pth
- Project_Resourse/models/at_3c2d_full_checkpoint.pth
- Project_Resourse/models/at_3c2d_fgsm_full_checkpoint.pth

FGSM checkpoint canonical state:
- at_3c2d_fgsm_full_checkpoint.pth has epoch_zero_based=31, best_epoch=27, best_robust_val_acc=74.329850.

---

## 5. Canonical Evaluation Artifacts

Stage 3 all-model fixed-set comparison (latest cycle):
- run_logs/stage3_fixed_eval_all_three_models_20260411_001931.log
- Project_Resourse/logs/fixed_adv_eval_all_three_models_20260411_001931.txt
- Project_Resourse/logs/stage3_complete_comparison_all3_20260411_001931.txt
- Project_Resourse/logs/fixed_adv_eval_all.txt

Stage 2 clean baseline (3C2D only):
- run_logs/stage2_eval_3c2d_clean_only_20260411_002438.log
- Project_Resourse/logs/stage2_eval_3c2d_clean_only_20260411_002438.txt

FGSM post-continuation fixed-set evaluation:
- run_logs/fixed_adv_eval_3c2d_fgsm_post32_20260410_210424.log
- Project_Resourse/logs/fixed_adv_eval_3c2d_fgsm.txt

---

## 6. Training Completeness and Curve Reconstruction Artifacts

- Project_Resourse/logs/3C2D_MaleX_clean_Baseline.txt
- Project_Resourse/logs/Resnet18_MaleX_clean_Baseline.txt
- Project_Resourse/logs/3C2D MaleX clean Baseline.txt
- Project_Resourse/logs/Resnet18 MaleX clean Baseline.txt
- Project_Resourse/logs/adversarial_training_log_3c2d_reconstructed.txt
- Project_Resourse/logs/adversarial_training_log_fgsm_reconstructed.txt
- Project_Resourse/logs/training_curve_3c2d_clean_baseline.png
- Project_Resourse/logs/training_curve_3c2d_fixed.png
- Project_Resourse/logs/training_curve_resnet18_malex_clean_baseline.png
- Project_Resourse/logs/adversarial_training_curve_3c2d.png
- Project_Resourse/logs/adversarial_training_curve_fgsm.png

---

## 7. Dataset and Fixed-Set Counts

MaleX split counts:
- Train total: 287,560
- Val total: 33,015
- Test total: 33,004
- Test benign: 17,153
- Test malware: 15,851

Fixed adversarial set counts:
- Project_Resourse/adversarial_test_set_malex/fgsm_eps0.05: 15,851 PNG
- Project_Resourse/adversarial_test_set_malex/pgd_eps0.05_steps40: 15,851 PNG
- Project_Resourse/adversarial_test_set/fgsm_eps0.05: 847 PNG
- Project_Resourse/adversarial_test_set/pgd_eps0.05_steps40: 847 PNG

---

## 8. Runtime Status and Remaining Work

Current runtime state:
- Stage 3 model triad is finalized and evaluated with full metrics on fixed adversarial sets.
- Stage 4 inference remains implemented and verified (/health and /compare).

Remaining tasks:
1) Resolve cross-split overlap risk and rerun diagnostics.
2) Package final Stage 4 thesis demo appendix bundle.
3) Keep commit scope tight around docs/code/log summaries, excluding large binaries.
