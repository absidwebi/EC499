# EC499 Folder Structure Overview

Last updated: 2026-04-25 (demo-prep cleanup sync)

This document reflects the current cleaned workspace layout used for Stage 4 demo and final report operations.

---

## 1. Top-Level Layout (Current)

EC499/
- AGENTS.md
- Agents.md
- EC499_Folder_Structure.md
- Project_Resourse/
- run_logs/
- ben_byteplot_imgs_zipped/
- mal_byteplot_imgs_zipped/
- mal_byteplot_imgs_zipped.tar.gz
- benign_images_256x256/
- benign_images_nataraj_v3/
- benign_pe_files/
- 2016904.2016908.pdf
- 2402.15267v1.pdf
- Adversarial_Robustness_in_Deep_Learning_Based_Malware_DetectionProjectProposal.pdf
- Adversarial_Robustness_Presentaion.pdf
- venv/

Commit hygiene:
- Keep dataset archives, model binaries, and large generated artifacts out of scoped documentation/code commits unless explicitly intended.

---

## 2. Active Python Scripts in Project_Resourse

Current `.py` files in `Project_Resourse/`:
- adversarial_train.py
- adversarial_train_fgsm.py
- app.py
- build_malex_split.py
- check_malex_hash_overlaps.py
- check_malex_labels.py
- check_malex_shuffle_sanity.py
- check_raw_png_sizes.py
- check_sizes.py
- collect_benign_pe.py
- config.py
- convert_to_malimg.py
- dataset_loader.py
- evaluate_attacks.py
- evaluate_attacks_fixed.py
- evaluate_base_models_testset.py
- fix_dataset_leakage.py
- fix_malex_split_overlaps.py
- fix_malex_val_test_overlaps.py
- fix_malimg_crossplit_duplicates.py
- generate_adversarial_test_set.py
- generate_malex_adv_testset.py
- inference.py
- models.py
- rebuild_dataset_1to1.py
- train.py
- train_3c2d.py
- train_resnet_pretrained.py
- verify_malex_source.py

Removed during cleanup (no longer present):
- train_efficientnet.py
- adversarial_train_efficientnet.py
- check_hash_overlaps.py
- split_benign_dataset.py
- verify_benign_collection.py
- verify_visual_duplicates.py
- and other archived/non-demo helper scripts removed from the active tree

---

## 3. Key Subdirectories Under Project_Resourse

- archive/
  - malex_dataset/ (active split root)
- logs/
  - training/evaluation logs and generated figure outputs
- models/
  - clean and adversarially trained checkpoints
- adversarial_test_set_malex/
  - primary fixed adversarial test subsets
- adversarial_test_set/
  - legacy smaller fixed adversarial subsets
- templates/
  - Stage 4 demo UI assets
- venv/
  - local Python environment scoped under `Project_Resourse`

---

## 4. Canonical Model Artifacts (Current)

- Project_Resourse/models/3c2d_malex_clean_vulnerable.pth
- Project_Resourse/models/3c2d_malex_adversarially_trained.pth
- Project_Resourse/models/3c2d_malex_fgsm_adversarially_trained.pth
- Project_Resourse/models/at_3c2d_full_checkpoint.pth
- Project_Resourse/models/at_3c2d_fgsm_full_checkpoint.pth

FGSM canonical checkpoint state:
- `at_3c2d_fgsm_full_checkpoint.pth` records epoch_zero_based=31, best_epoch=27, best_robust_val_acc=74.329850.

---

## 5. Data and Evaluation Baselines

MaleX split totals (unchanged):
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

## 6. Demo-Prep Runtime Notes

Current state:
- Stage 3 clean/PGD-AT/FGSM-AT comparison remains finalized.
- Stage 4 inference code path remains present (`inference.py`, `app.py`, `templates/index.html`, `Dockerfile`).
- Workspace was cleaned to reduce non-essential scripts/docs before demo.

Operational caution:
- Source repo currently includes multiple tracked deletions/modifications from cleanup; when committing, stage only intended files.

---

## 7. Mirror and Documentation Sync

Canonical structure/context docs:
- Source repo: `EC499_Folder_Structure.md`, `Project_Resourse/PROJECT_CONTEXT.md`, `Project_Resourse/CURRENT_STATE.md`, `Project_Resourse/MASTER_CONTEXT.md`
- Report repo mirrors: `EC499_Folder_Structure.md`, `PROJECT_CONTEXT.md`, `CURRENT_STATE.md`, `MASTER_CONTEXT.md`

For this update cycle:
- `EC499_Folder_Structure.md` is synchronized to both repos to match demo-cleaned workspace state.
