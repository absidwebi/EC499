# CURRENT_STATE.md
# EC499 - Current Project State

Last updated: 2026-04-04

---

## 1. Stage Progress Snapshot

| Stage | Status | Notes |
|---|---|---|
| Stage 1 - Dataset Preparation | DONE | MaleX split pipeline is active under archive/malex_dataset |
| Stage 2 - Base Model Selection | DONE | 3C2D selected from final test-set comparison |
| Stage 3 Part 1 - Attack Evaluation | DONE | Clean baseline vulnerability confirmed |
| Stage 3 Part 2 - Adversarial Training | IN PROGRESS (continuation run) | PGD resume run active toward 50 epochs; best robust val currently 74.12% |
| Stage 3 Part 2b - FGSM Defense Branch | DONE | 20-epoch FGSM run completed with checkpoint + curve |
| Stage 4 - Inference API + Demo | IMPLEMENTED + VALIDATED (with residual caveat) | inference.py, app.py, UI, Dockerflow verified in session; isolated-network host reachability caveat remains |

---

## 2. What Changed Since 2026-04-02

### 2.1 Python script changes in this window

1) Updated: Project_Resourse/adversarial_train.py
- Changed `NUM_EPOCHS` from `20` to `50`.
- Reason: continue from saved checkpoint instead of truncating at epoch cap 20.

No other tracked `.py` files changed in this update window.

### 2.2 New run outputs and comparison artifacts

- run_logs/evaluate_attacks_3c2d_post35.log
- run_logs/evaluate_attacks_3c2d_at_post35.log
- Project_Resourse/logs/attack_evaluation_results_3c2d_at.txt
- Project_Resourse/logs/attack_comparison_3c2d_clean_vs_at_post35.txt

### 2.3 Existing major scripts/artifacts still in active use (from prior cycle)

1) Updated: Project_Resourse/config.py
- Added constants for fixed adversarial test set and FGSM-trained model path.

2) New: Project_Resourse/generate_malex_adv_testset.py
- One-time generator + verifier for deterministic MaleX adversarial test subsets.

3) New: Project_Resourse/evaluate_attacks_fixed.py
- Evaluates clean/PGD/FGSM models on fixed subsets for reproducible comparison.

4) New: Project_Resourse/adversarial_train_fgsm.py
- FGSM adversarial training with full resume checkpoints and per-epoch logs/curves.

5) New: Project_Resourse/inference.py
- Static PE validation + Nataraj conversion + OpenCV resize + CPU inference pipeline.

6) New: Project_Resourse/app.py
- Flask API with /health, /predict, 50MB cap, tempfile handling, JSON error paths.

### 2.4 New major artifacts (cumulative)

- Project_Resourse/adversarial_test_set_malex/
- Project_Resourse/models/3c2d_malex_fgsm_adversarially_trained.pth
- Project_Resourse/models/at_3c2d_fgsm_full_checkpoint.pth
- Project_Resourse/models/at_3c2d_full_checkpoint.pth
- Project_Resourse/logs/fixed_adv_eval_3c2d.txt
- Project_Resourse/logs/fixed_adv_eval_3c2d_pgd.txt
- Project_Resourse/logs/fixed_adv_eval_all.txt
- Project_Resourse/logs/adversarial_training_log_fgsm.txt
- Project_Resourse/logs/adversarial_training_curve_fgsm.png
- Project_Resourse/logs/adversarial_training_curve_3c2d.png

---

## 3. Current Quantitative State

### 3.1 PGD defense continuation (active)

From current full-checkpoint metadata:
- epoch_zero_based: 35
- resume_next_epoch_1based: 37
- best_robust_val_acc: 74.1239%
- best_epoch: 35
- epochs_no_improve: 1

Current state summary:
- PGD continuation run is still active in terminal and writing to canonical run log.
- Latest completed summary in canonical log: Epoch 36/50 with Val Clean 80.11% and Val Robust 73.97%.
- Model and full checkpoint are updating under models/.

### 3.2 FGSM defense branch (complete)

Final FGSM epoch log snapshot:
- Epoch 20/20
- Train Loss: 0.4971
- Train Acc: 72.73%
- Val Clean: 77.37%
- Val Robust (FGSM): 71.68%
- Best robust val acc (checkpoint): 72.7306% at epoch 19

### 3.3 Clean vs AT mirrored attack comparison (post-epoch-35 checkpoint)

From Project_Resourse/logs/attack_comparison_3c2d_clean_vs_at_post35.txt:
- Clean model:
  - Clean acc 85.29%
  - FGSM e=0.10 acc 3.49%
  - PGD e=0.05 (40 steps) acc 0.62%
- AT (PGD) model:
  - Clean acc 80.03%
  - FGSM e=0.10 acc 71.39%
  - PGD e=0.05 (40 steps) acc 74.22%

Interpretation:
- AT model delivers major robustness gains under identical attack settings, with expected clean-accuracy tradeoff.

---

## 4. Stage 4 Inference and Demo Status

Implemented and locally validated:
- Project_Resourse/inference.py
- Project_Resourse/app.py
- Project_Resourse/templates/index.html
- Project_Resourse/Dockerfile

Local endpoint tests passed:
- GET /health returned status/model json.
- POST /predict with valid PE returned label/confidence/logit/image_b64.
- POST /predict with non-PE returned ValueError-backed JSON error.

Additional note:
- Docker is available on this host now (`docker --version` reports 28.2.2).
- Latest Stage 4 verification notes include one residual caveat around host reachability behavior under isolated `--network=none` conditions.

---

## 5. Current Problems / Blockers

1) Split-overlap risk remains open (scientific methodology risk).
2) PGD continuation run is still in progress to epoch 50, so final continuation metrics are not locked yet.
3) Stage 4 has a residual reproducibility caveat under isolated-network host reachability behavior.
4) Large unrelated local untracked files remain in workspace and should stay out of scoped commits.

---

## 6. Immediate Next Actions

1) Let active PGD continuation run finish and record final best robust-val metric.
2) Re-run clean-vs-AT mirrored attack comparison on final checkpoint.
3) Resolve split-overlap issue and rerun overlap integrity diagnostics.
4) Archive final Stage 4 smoke-test evidence with command outputs for thesis appendix traceability.
