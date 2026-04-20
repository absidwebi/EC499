# EC499 - Current Project State

Last updated: 2026-04-20 (source/report context sync after figure finalization cycle)

---

## 1. Stage Progress Snapshot

| Stage | Status | Notes |
|---|---|---|
| Stage 1 - Dataset Preparation | DONE | MaleX split pipeline active in archive/malex_dataset |
| Stage 2 - Base Model Selection | DONE | 3C2D selected; clean baseline metrics re-run and archived |
| Stage 3 Part 1 - Attack Evaluation | DONE | Clean vulnerability confirmed |
| Stage 3 Part 2 - PGD Adversarial Training | DONE | Early stop epoch 40/50, best robust 74.123883% at epoch 35 |
| Stage 3 Part 2b - FGSM Adversarial Training | DONE | Continued to epoch 32 (from epoch 20 resume), best robust 74.329850% at epoch 27 |
| Stage 3 Fixed-Set 3-Model Comparison | DONE | Fresh all-model run generated on 2026-04-11 |
| Stage 4 - Inference API + Demo | IMPLEMENTED + VERIFIED | /predict and /compare validated; bridge network with -p required |

---

## 2. Canonical Checkpoints and Training State

3C2D clean baseline:
- Project_Resourse/models/3c2d_malex_clean_vulnerable.pth

3C2D AT (PGD):
- Project_Resourse/models/3c2d_malex_adversarially_trained.pth
- Project_Resourse/models/at_3c2d_full_checkpoint.pth
- Best robust val acc: 74.123883% at epoch 35

3C2D AT (FGSM):
- Project_Resourse/models/3c2d_malex_fgsm_adversarially_trained.pth
- Project_Resourse/models/at_3c2d_fgsm_full_checkpoint.pth
- Checkpoint state: epoch_zero_based=31, best_epoch=27, best_robust_val_acc=74.329850

Canonical FGSM decision:
- The canonical FGSM checkpoint is the continued run (best epoch 27), not the earlier epoch-19 state.

---

## 3. Latest Evaluation Artifacts (Finalized in This Cycle)

Stage 3 all-model fixed-set run:
- run_logs/stage3_fixed_eval_all_three_models_20260411_001931.log
- Project_Resourse/logs/fixed_adv_eval_all_three_models_20260411_001931.txt
- Project_Resourse/logs/stage3_complete_comparison_all3_20260411_001931.txt

Stage 2 clean-only 3C2D run:
- run_logs/stage2_eval_3c2d_clean_only_20260411_002438.log
- Project_Resourse/logs/stage2_eval_3c2d_clean_only_20260411_002438.txt

Reconstructed training completeness artifacts:
- Project_Resourse/logs/3C2D_MaleX_clean_Baseline.txt
- Project_Resourse/logs/Resnet18_MaleX_clean_Baseline.txt
- Project_Resourse/logs/adversarial_training_log_3c2d_reconstructed.txt
- Project_Resourse/logs/adversarial_training_log_fgsm_reconstructed.txt
- Project_Resourse/logs/training_curve_3c2d_clean_baseline.png
- Project_Resourse/logs/training_curve_resnet18_malex_clean_baseline.png

---

## 4. Latest 3-Model Metrics Summary (Fixed Adversarial Set)

| Model | Clean Accuracy | FGSM Recall | FGSM Evasion | PGD Recall | PGD Evasion |
|---|---:|---:|---:|---:|---:|
| 3C2D Clean | 85.2927% | 15.29% | 84.71% | 0.53% | 99.47% |
| 3C2D AT (PGD) | 80.0297% | 74.97% | 25.03% | 74.94% | 25.06% |
| 3C2D AT (FGSM) | 77.9633% | 68.92% | 31.08% | 68.81% | 31.19% |

---

## 5. What Changed In This Update Window

Updated script:
- Project_Resourse/adversarial_train_fgsm.py

Added behavior:
- FGSM_NUM_EPOCHS env override
- FGSM_EARLY_STOP_PATIENCE env override
- FGSM_RESUME_FROM_BEST_WEIGHTS resume mode
- Resume flow now supports continuation from checkpoint epoch state while reloading best epoch weights

Why this mattered:
- Required to continue FGSM from epoch 20, use best epoch-19 weights, append to canonical run log, and stop only when improvement ceased.

---

## 6. Open Problems / Unchecked Requirements

1) Cross-split overlap risk remains open and must be closed before final thesis claims.
2) Stage 4 committee/demo appendix packaging is still pending.
3) evaluate_base_models_testset.py relative output path is brittle when invoked from Project_Resourse cwd.
4) Sidecar metadata strategy should be standardized across defended artifacts.
5) Large unrelated untracked binaries exist in workspace and must remain outside scoped commits.

---

## 7. Immediate Next Actions

1) Keep report tables locked to the 2026-04-11 canonical artifacts.
2) If additional attack tables are needed, run only against canonical checkpoints.
3) Close overlap risk and rerun overlap diagnostics.
4) Package Stage 4 appendix with reproducible command/output evidence.

---

## 8. Delta State (2026-04-11 -> 2026-04-20)

### 8.1 Source repository (`EC499`, branch `stage4-demo`)

Tracked changes in this window:
- `Project_Resourse/Dockerfile` updated to pin `pefile==2023.2.7`.
- `Project_Resourse/requirements_inference.txt` updated to pin `pefile==2023.2.7`.
- No tracked `.py` file changes were committed in this window.

Interpretation:
- Core model-training/evaluation logic is stable relative to the 2026-04-11 canonical metrics.
- This window focused on deployment reproducibility and report figure quality.

### 8.2 Report repository (`EC499-Report`, branch `main`)

Completed report updates:
- Added and refined architecture/attack/deployment figures.
- Added and iteratively corrected t-SNE figure until final layout requirements were met.

Final t-SNE status:
- Final artifact: `Project_Resourse/logs/report_figures/tsne_embedding_3c2d.png`
- "PGD attack ->" overlay text is fully removed.
- Cluster-separation annotation text removed from panel stats.
- Latest commit carrying the finalized text-removal state: `c8a4819`.

### 8.3 Current known issues (active)

1) Cross-split overlap risk remains unresolved.
2) Stage 4 final appendix packaging remains pending.
3) `evaluate_base_models_testset.py` output path remains CWD-sensitive.
4) Source working tree currently has tracked local modifications outside this sync task:
	`Project_Resourse/base_model_testset_results.json`,
	`Project_Resourse/models/3c2d_malex_fgsm_adversarially_trained.pth`,
	`Project_Resourse/templates/index.html`.
5) Figure-generation helper scripts in this cycle were `/tmp/*.py` operational scripts (not tracked in repo).

### 8.4 Immediate execution priorities

1) Resolve overlap diagnostics and refresh dataset-integrity evidence.
2) Freeze Stage 4 demo package contents (commands, sample files, expected outputs).
3) Keep source commits narrowly scoped and avoid mixing report-only artifact churn into pipeline code commits.
