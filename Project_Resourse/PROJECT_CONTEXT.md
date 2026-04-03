# PROJECT_CONTEXT.md
# EC499 - Adversarial Robustness in Deep Learning-Based Malware Detection

Student: Abdulsalam Ashraf Aldwebi (ID: 2210245306)
Supervisor: Dr. Suad Elgeder | University of Tripoli
GitHub: https://github.com/absidwebi/EC499
Primary Machine: Ubuntu, RTX 4060, /home/alucard-00/EC499/
Python Environment (active and verified): /home/alucard-00/EC499/Project_Resourse/venv/

Last updated: 2026-04-04

---

## 1. Project Objective

Build a reproducible malware image classification pipeline and evaluate adversarial vulnerability and defense under controlled methodology, then deliver a static-analysis inference demo.

Binary task:
- Benign = 0
- Malware = 1

Core rule set maintained in active scripts:
- Single-logit output + BCEWithLogitsLoss
- Adversarial clamp range [-1.0, 1.0]
- No geometric augmentation in byteplot pipeline
- num_workers=0 for Linux CUDA stability

---

## 2. Active Data Track

Current active branch for Stage 2/3/4 is MaleX byteplot data.

Source roots:
- Benign: /home/alucard-00/EC499/ben_byteplot_imgs_zipped/byteplot_imgs_RxR/256/
- Malware: /home/alucard-00/EC499/mal_byteplot_imgs_zipped/byteplot_imgs_RxR/256/

Working split root:
- /home/alucard-00/EC499/Project_Resourse/archive/malex_dataset/

MaleX split sizes (current loader output):
- Train: 287,560 (143,780 benign + 143,780 malware)
- Val: 33,015
- Test: 33,004

Methodology note:
- Cross-split overlap remediation is still tracked as an open scientific risk item.

---

## 3. Stage 2 Final Selection (Complete)

Final clean baseline candidates:
- MaleX3C2D
- ResNet-18 pretrained grayscale variant

Authoritative artifact:
- Project_Resourse/base_model_testset_results.json

Selected Stage 3 baseline:
- MaleX3C2D

Reason:
- Slight edge on accuracy and malware F1 while overall macro quality remained nearly tied.

---

## 4. Stage 3 Defense Track (Current)

### 4.1 Stage 3 part 1 (clean vulnerability): complete

Script:
- Project_Resourse/evaluate_attacks.py

Artifact:
- Project_Resourse/logs/attack_evaluation_results_3c2d.txt

Clean baseline collapse under stronger attacks remains confirmed.

### 4.2 Stage 3 part 2 (PGD adversarial training): continuation run active toward 50 epochs

Primary run log:
- run_logs/adversarial_train_ 3C2D_Fixed_malex_stage3.log

Current runtime state (latest check):
- PGD adversarial training process is active.
- Saved full-checkpoint state currently shows:
  - epoch_zero_based: 35
  - resume_next_epoch_1based: 37
  - best_robust_val_acc: 74.1239%
  - best_epoch: 35
  - epochs_no_improve: 1

Latest completed epoch summary in canonical run log:
- Epoch 36/50
- Train Loss: 0.4713
- Train Acc: 74.75%
- Val Clean: 80.11%
- Val Robust: 73.97%

Checkpoint and model artifacts:
- Project_Resourse/models/at_3c2d_full_checkpoint.pth
- Project_Resourse/models/3c2d_malex_adversarially_trained.pth (updated during continuation)
- Project_Resourse/logs/adversarial_training_log_3c2d.txt
- Project_Resourse/logs/adversarial_training_curve_3c2d.png

Interpretation:
- The resumed PGD run moved well beyond the earlier 64.91% robust-val baseline and reached 74.12% best robust validation.
- Epoch 36 did not improve best robust validation, but training is still active and progressing through epoch 37 at update time.

### 4.3 Fixed adversarial test set + deterministic comparison: complete

Fixed-set generator/output:
- Project_Resourse/generate_malex_adv_testset.py
- Project_Resourse/adversarial_test_set_malex/
  - fgsm_eps0.05/images: 15,851
  - pgd_eps0.05_steps40/images: 15,851

Fixed-set evaluator:
- Project_Resourse/evaluate_attacks_fixed.py

All-model fixed-set comparison artifact:
- run_logs/fixed_adv_eval_all_models_final.log
- Project_Resourse/logs/fixed_adv_eval_all.txt

Selected deterministic results:
- Clean model:
  - Clean accuracy: 85.29%
  - FGSM fixed recall: 15.29%
  - PGD fixed recall: 0.53%
- PGD-defended model:
  - Clean accuracy: 73.07%
  - FGSM fixed recall: 58.56%
  - PGD fixed recall: 58.29%
- FGSM-defended model:
  - Clean accuracy: 77.33%
  - FGSM fixed recall: 67.97%
  - PGD fixed recall: 67.84%

Interpretation:
- Deterministic fixed-set evaluation confirms substantial robustness gain over clean baseline for both defended models.

### 4.4 FGSM adversarial training branch: complete

Script:
- Project_Resourse/adversarial_train_fgsm.py

Artifacts:
- Project_Resourse/models/3c2d_malex_fgsm_adversarially_trained.pth
- Project_Resourse/models/at_3c2d_fgsm_full_checkpoint.pth
- Project_Resourse/logs/adversarial_training_log_fgsm.txt
- Project_Resourse/logs/adversarial_training_curve_fgsm.png

Final checkpoint metadata:
- epoch_zero_based: 19 (20 epochs complete)
- best_robust_val_acc: 72.7306%
- best_epoch: 19

### 4.5 Post-epoch-35 clean vs AT attack comparison: complete

Clean-model attack evaluation (no rerun during AT pass):
- run_logs/evaluate_attacks_3c2d_post35.log
- Project_Resourse/logs/attack_evaluation_results_3c2d.txt

AT-model mirrored evaluation (same FGSM/PGD settings):
- run_logs/evaluate_attacks_3c2d_at_post35.log
- Project_Resourse/logs/attack_evaluation_results_3c2d_at.txt

Side-by-side delta report:
- Project_Resourse/logs/attack_comparison_3c2d_clean_vs_at_post35.txt

Key outcomes:
- Clean accuracy: 85.29% (clean model) vs 80.03% (AT model), delta -5.26 pp
- PGD (e=0.05, steps=40): 0.62% (clean model) vs 74.22% (AT model), delta +73.60 pp
- FGSM (e=0.10): 3.49% (clean model) vs 71.39% (AT model), delta +67.90 pp

Interpretation:
- Robustness gains are large and consistent across all attack strengths.
- Clean-accuracy tradeoff remains present and expected.

---

## 5. Stage 4 Inference and Deployment (Implemented + Verified with Remaining Caveat)

### 5.1 Implemented files

- Project_Resourse/inference.py
  - Static PE validation via pefile (no execution), raw-byte read, Nataraj conversion, OpenCV INTER_AREA resize, normalization to [-1,1], CPU-only model inference.
- Project_Resourse/app.py
  - Flask API with /health, /, /predict, upload limit, secure tempfile handling, JSON error mapping.
- Project_Resourse/templates/index.html
  - Offline single-file UI with upload flow, spinner, result card, confidence bar, byteplot display, reset flow.
- Project_Resourse/Dockerfile
  - Container recipe for CPU inference demo.

### 5.2 Validation status

Local API checks passed:
- /health returned model status json.
- /predict with valid PE returned benign/malware label, confidence, logit, and non-empty image_b64.
- /predict with non-PE returned validation-error json (400 path).

Container-level checks were executed in the latest verification cycle with a single environment caveat:
- isolated-mode (`--network=none`) host reachability behavior in this setup remains non-standard and should be considered when reproducing the exact demo flow.

### 5.3 Current Stage 4 residual requirement

- Re-run one final Stage 4 end-to-end smoke pass (health + valid PE + invalid PE + UI + dockerized run) after Stage 3 training stabilizes, and record outputs in a dedicated run log artifact for thesis appendix traceability.

---

## 6. Code Changes Since Last Context Update

### 6.1 Updated existing script

1) Project_Resourse/config.py
- Added Stage 3 fixed-set and FGSM model-path constants:
  - MALEX_ADV_TEST_SET_DIR / MALEX_ADV_TEST_SET_DIR_STR
  - MALEX_3C2D_FGSM_ADV_MODEL_PATH_STR

Why:
- Standardize paths for fixed adversarial set and FGSM defense artifacts across scripts.

### 6.2 New scripts

2) Project_Resourse/generate_malex_adv_testset.py
- Generates fixed adversarial PNG test subsets from malware test samples using clean 3C2D attacker.
- Includes verify mode to validate counts, labels, image mode/size.

Why:
- Remove stochastic variance from repeated on-the-fly PGD generation and enforce deterministic comparisons.

3) Project_Resourse/evaluate_attacks_fixed.py
- Evaluates clean, PGD-defended, and FGSM-defended models on fixed adversarial subsets.
- Produces deterministic model-comparison logs in run_logs and logs.

Why:
- Establish reproducible clean-vs-defended benchmarking on identical adversarial pixels.

4) Project_Resourse/adversarial_train_fgsm.py
- Parallel FGSM defense training pipeline with resume checkpointing, early stopping, and curve generation.

Why:
- Build a second defense baseline under same checkpointing and reporting framework.

5) Project_Resourse/inference.py
- Full static PE-to-prediction inference pipeline aligned with training preprocessing.

Why:
- Stage 4 deliverable: production-style model inference entry point.

6) Project_Resourse/app.py
- Flask API and upload workflow around inference engine.

Why:
- Stage 4 deploy/demo requirement for committee-facing interaction.

### 6.3 Additional script change after 2026-04-02

7) Project_Resourse/adversarial_train.py
- Updated `NUM_EPOCHS` from `20` to `50`.

Why:
- Continue PGD adversarial training from existing checkpoint beyond the previous cap, preserving optimizer/scheduler state and append-only logging.

Note on tracked Python changes in this cycle:
- No other tracked `.py` files changed after this update window.

---

## 7. Known Open Items and Risks

1) MaleX split-overlap methodological risk remains unresolved.
2) PGD continuation run to epoch 50 is still active; final continuation metrics are not yet frozen.
3) Stage 4 has a residual reproducibility caveat around isolated-network host reachability behavior.
4) Workspace contains large unrelated untracked binaries/archives/generated images that must stay out of scoped commits.

---

## 8. Immediate Next Actions

1) Let active PGD continuation run finish to epoch 50 and capture final robust-val/best-epoch metrics.
2) Re-run side-by-side clean vs AT attack comparison on the latest final checkpoint.
3) Resolve split-overlap risk and rerun integrity checks for thesis-grade claims.
4) Archive Stage 4 final smoke-test evidence in a reproducible run log bundle.
