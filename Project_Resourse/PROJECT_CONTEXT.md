# EC499 - Adversarial Robustness in Deep Learning-Based Malware Detection

Student: Abdulsalam Ashraf Aldwebi (ID: 2210245306)
Supervisor: Dr. Suad Elgeder | University of Tripoli
GitHub: https://github.com/absidwebi/EC499
Primary machine: Ubuntu Linux, RTX 4060, /home/alucard-00/EC499
Primary interpreter for current cycle: /home/alucard-00/EC499/venv/bin/python

Last updated: 2026-04-07 (documentation synchronization + Stage 3 metric/provenance code upgrade)

---

## 1. Project Objective

Build a reproducible malware-image binary classifier and rigorously evaluate adversarial vulnerability and defense for a thesis-grade study, then expose the defended-vs-clean behavior through a static-analysis Stage 4 inference demo.

Binary task:
- Benign = 0
- Malware = 1

Method constraints actively enforced in pipeline code:
- Single-logit output and BCEWithLogitsLoss
- Adversarial clamp range [-1.0, 1.0]
- No geometric augmentation in byteplot pipeline
- num_workers = 0 for Linux/CUDA stability
- Centralized paths from config.py (avoid hardcoded machine paths in active scripts)

---

## 2. Active Data Track (MaleX Byteplot)

Source roots:
- Benign: /home/alucard-00/EC499/ben_byteplot_imgs_zipped/byteplot_imgs_RxR/256/
- Malware: /home/alucard-00/EC499/mal_byteplot_imgs_zipped/byteplot_imgs_RxR/256/

Working split root:
- /home/alucard-00/EC499/Project_Resourse/archive/malex_dataset/

Current verified split sizes:
- Train total: 287,560
  - Train benign: 143,780
  - Train malware: 143,780
- Val total: 33,015
- Test total: 33,004
  - Test benign: 17,153
  - Test malware: 15,851

Open scientific risk still tracked:
- Cross-split overlap remediation is not closed yet and remains a thesis-quality blocker for final claims.

---

## 3. Stage 2 Baseline Selection (Locked)

Final clean candidates:
- MaleX3C2D
- ResNet-18 pretrained grayscale

Authoritative artifact:
- Project_Resourse/base_model_testset_results.json

Selected Stage 3 baseline:
- MaleX3C2D

Selection rationale:
- Slight edge on test-set accuracy and malware-focused F1 with comparable macro behavior.

---

## 4. Stage 3 Defense Track (Current Canon)

### 4.1 Part 1 - Clean vulnerability (complete)

Script:
- Project_Resourse/evaluate_attacks.py

Artifact:
- Project_Resourse/logs/attack_evaluation_results_3c2d.txt

Outcome:
- Clean baseline collapses under stronger FGSM/PGD settings, validating need for adversarial training.

### 4.2 Part 2 - PGD adversarial training (complete)

Canonical run log:
- run_logs/adversarial_train_ 3C2D_Fixed_malex_stage3.log

Canonical per-epoch text log:
- Project_Resourse/logs/adversarial_training_log_3c2d.txt

Final run outcome:
- Early stopping at epoch 40/50
- Best robust validation accuracy: 74.123883% at epoch 35

Primary artifacts:
- Project_Resourse/models/3c2d_malex_adversarially_trained.pth
- Project_Resourse/models/at_3c2d_full_checkpoint.pth
- Project_Resourse/logs/adversarial_training_curve_3c2d.png

Checkpoint values currently confirmed from at_3c2d_full_checkpoint.pth:
- epoch (zero-based): 39
- best_robust_val_acc: 74.12388308344691
- best_epoch: 35

Important provenance note:
- Code now supports writing a best-weights sidecar metadata JSON.
- Existing historical best weights were produced before this sidecar existed, so the metadata file is not yet present for current artifacts.

### 4.3 Part 2b - FGSM defense branch (complete)

Artifacts:
- Project_Resourse/models/3c2d_malex_fgsm_adversarially_trained.pth
- Project_Resourse/models/at_3c2d_fgsm_full_checkpoint.pth
- Project_Resourse/logs/adversarial_training_log_fgsm.txt

Reference summary:
- best_robust_val_acc: 72.7306%
- best_epoch: 19

### 4.4 Deterministic fixed adversarial comparison (complete)

Primary fixed-set path used for MaleX-scale comparison:
- Project_Resourse/adversarial_test_set_malex/
  - fgsm_eps0.05: 15,851 PNG images
  - pgd_eps0.05_steps40: 15,851 PNG images

Legacy small fixed set still exists (not primary for final report):
- Project_Resourse/adversarial_test_set/
  - fgsm_eps0.05: 847 PNG images
  - pgd_eps0.05_steps40: 847 PNG images

Evaluator:
- Project_Resourse/evaluate_attacks_fixed.py

Output artifact:
- Project_Resourse/logs/fixed_adv_eval_all.txt

### 4.5 Clean vs AT mirrored attack comparison (post-epoch-35 reference)

Artifacts:
- Project_Resourse/logs/attack_evaluation_results_3c2d.txt
- Project_Resourse/logs/attack_evaluation_results_3c2d_at.txt
- Project_Resourse/logs/attack_comparison_3c2d_clean_vs_at_post35.txt

Reference deltas:
- Clean (no attack): clean 85.29% vs AT 80.03% (delta -5.26 pp)
- FGSM e=0.10: clean 3.49% vs AT 71.39% (delta +67.90 pp)
- PGD e=0.05 steps=40: clean 0.62% vs AT 74.22% (delta +73.60 pp)

Interpretation:
- Strong and consistent robustness gains under attack with expected clean-accuracy tradeoff.

---

## 5. Stage 4 Inference and Deployment (Verified Scope)

### 5.1 Implemented deployment paths

Core files:
- Project_Resourse/inference.py
- Project_Resourse/app.py
- Project_Resourse/templates/index.html
- Project_Resourse/Dockerfile

Active endpoints:
- GET /
- GET /health
- POST /predict
- POST /compare

### 5.2 Runtime checks confirmed in this cycle

- /health returns status JSON
- /compare rejects non-PE inputs with HTTP 400
- /compare accepts valid PE and returns HTTP 200 with comparison payload keys:
  - byteplot_b64
  - clean_256_b64
  - adv_256_b64
  - clean_model
  - at_model
  - attack_params

### 5.3 Docker networking clarification (important)

Empirical behavior in this environment:
- Using --network=none with -p did not provide browser reachability.
- Bridge networking with -p 5000:5000 was required for local UI access.

Reproducibility interpretation:
- Current Docker setup is reproducible for Stage 4 inference/demo delivery.
- It is not a full end-to-end training reproducibility package yet.

---

## 6. Code Changes in Current Update Window (Python)

### 6.1 Project_Resourse/evaluate_base_models_testset.py

Upgrades:
- Added full confusion-style metric suite helper
- Added threshold metrics (TPR at constrained FPR)
- Added balanced accuracy, MCC, per-class precision/recall, specificity, FPR/FNR
- Expanded console and JSON reporting payload

Why:
- Stage 2 model selection and thesis tables require richer statistical evidence than accuracy-only output.

### 6.2 Project_Resourse/evaluate_attacks.py

Upgrades:
- Refactored evaluate() to return full metrics dictionary instead of only accuracy
- Added per-setting metrics for clean, FGSM, and PGD evaluations
- Added detailed metrics section in output log while preserving summary table

Why:
- Stage 3 vulnerability analysis now captures class-level behavior and threshold-sensitive robustness quality.

### 6.3 Project_Resourse/evaluate_attacks_fixed.py

Upgrades:
- Added full clean-test metric block for each model
- Retained malware-only deterministic adversarial recall/evasion comparison logic
- Added threshold metrics and AUC-safe handling

Why:
- Deterministic fixed-set evaluation now includes complete clean baseline quality context for defended-vs-clean interpretation.

### 6.4 Project_Resourse/adversarial_train.py

Upgrades:
- Added write_best_weights_metadata() helper for sidecar provenance JSON
- Added backfill behavior during resume when historical best weights exist without sidecar
- Metadata schema includes:
  - model identity
  - best epoch and robust score
  - clean-model source path
  - attack config
  - run config and UTC update time

Why:
- Improve experiment traceability and checkpoint provenance for thesis and audit requirements.

---

## 7. Known Open Items and Risks

1) Cross-split overlap remediation is still not finalized.
2) One final mirrored attack-evaluation pass using locked reporting protocol is still recommended after metric-script upgrades.
3) Stage 4 committee demo package is not yet fully assembled (fixed sample list + reproducible command/output bundle).
4) Workspace contains many large untracked binaries and archives; these must stay out of normal source commits.
5) Best-weights sidecar metadata file is not present yet for existing PGD best artifact (feature is now implemented in code).

---

## 8. Immediate Next Actions

1) Run final audit-grade evaluation pass with upgraded metric scripts and archive dated artifacts.
2) Resolve overlap risk and rerun overlap diagnostics before thesis finalization.
3) Produce Stage 4 appendix bundle:
   - one benign PE, one malware PE, one invalid file
   - exact commands
   - expected API outputs and screenshots
4) Materialize sidecar metadata for locked best checkpoint by resume/backfill workflow or controlled metadata generation pass.
