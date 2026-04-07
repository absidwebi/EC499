# EC499 - Current Project State

Last updated: 2026-04-07 (doc sync + Stage 3 metrics/provenance code update)

---

## 1. Stage Progress Snapshot

| Stage | Status | Notes |
|---|---|---|
| Stage 1 - Dataset Preparation | DONE | MaleX split pipeline active in archive/malex_dataset |
| Stage 2 - Base Model Selection | DONE | 3C2D selected |
| Stage 3 Part 1 - Attack Evaluation | DONE | Vulnerability confirmed |
| Stage 3 Part 2 - PGD Adversarial Training | DONE | Early stop epoch 40/50, best robust 74.123883% at epoch 35 |
| Stage 3 Part 2b - FGSM Branch | DONE | Best robust 72.7306% at epoch 19 |
| Stage 4 - Inference API + Demo | IMPLEMENTED + VERIFIED | /predict and /compare active |

---

## 2. Verified Quantitative Baseline

Dataset split counts:
- Train: 287,560 (143,780 benign + 143,780 malware)
- Val: 33,015
- Test: 33,004 (17,153 benign + 15,851 malware)

Primary fixed adversarial test set (MaleX-scale):
- Project_Resourse/adversarial_test_set_malex/fgsm_eps0.05 -> 15,851 PNG
- Project_Resourse/adversarial_test_set_malex/pgd_eps0.05_steps40 -> 15,851 PNG

PGD checkpoint summary:
- at_3c2d_full_checkpoint.pth exists
- checkpoint epoch (zero-based): 39
- best_robust_val_acc: 74.12388308344691
- best_epoch: 35

---

## 3. What Changed In This Update Window

Updated scripts:
- Project_Resourse/evaluate_base_models_testset.py
- Project_Resourse/evaluate_attacks.py
- Project_Resourse/evaluate_attacks_fixed.py
- Project_Resourse/adversarial_train.py

Change set summary:
- Added full confusion-style metrics and threshold metrics (TPR@FPR constraints).
- Added class-wise precision/recall, specificity, FPR/FNR, balanced accuracy, MCC, AUC guards.
- Expanded attack logs with detailed per-setting metric blocks.
- Added best-checkpoint sidecar metadata writer plus resume backfill path.

Current artifact reality:
- Sidecar metadata capability is in code.
- Existing PGD best weight artifact still has no sidecar file yet.

---

## 4. Stage 4 Deployment State (Clarified)

Validated behavior:
- /health responds correctly.
- /compare returns 400 on invalid non-PE input.
- /compare returns 200 with full payload on valid PE input.

Networking clarification from direct testing:
- --network=none with -p was not browser-reachable in this environment.
- Bridge networking with -p is required for localhost demo access.

Interpretation:
- Docker setup is currently sufficient for reproducible Stage 4 inference/demo delivery.
- Full training reproducibility across new machines still requires separate environment and data regeneration protocol.

---

## 5. Open Problems / Unchecked Requirements

1) Cross-split overlap risk still open.
2) One final audit pass with updated metric scripts is still pending.
3) Stage 4 committee demo bundle is not packaged yet.
4) Sidecar provenance file for current best PGD artifact is not materialized yet.
5) Large unrelated untracked files remain in workspace and must stay outside scoped commits.

---

## 6. Immediate Next Actions

1) Run final evaluation bundle and archive dated logs/results.
2) Close split-overlap risk and rerun verification scripts.
3) Build reproducible Stage 4 demo appendix package (samples + commands + outputs).
4) Generate/checkpoint best-weights sidecar metadata for final defended model provenance.
