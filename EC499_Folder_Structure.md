# EC499 Folder Structure Overview (Updated)

Last updated: 2026-03-29

This is the current high-level layout of the EC499 workspace. Large datasets and generated artifacts are summarized.

## Directory Tree (high level)

```text
EC499/
├── 2016904.2016908.pdf
├── 2402.15267v1.pdf
├── Adversarial_Robustness_Presentaion.pdf
├── Adversarial_Robustness_in_Deep_Learning_Based_Malware_DetectionProjectProposal.pdf
├── AGENTS.md
├── archive.zip
├── ben_byteplot_imgs_zipped/
│   └── byteplot_imgs_RxR/
│       └── 256/
├── benign_images_256x256/
├── benign_images_nataraj_v3/
├── benign_images_test/
├── mal_byteplot_imgs_zipped/
│   └── byteplot_imgs_RxR/
│       └── 256/
├── Project_Resourse/
│   ├── adversarial_test_set/
│   │   ├── fgsm_eps0.05/
│   │   │   └── images/
│   │   └── pgd_eps0.05_steps40/
│   │       └── images/
│   ├── archive/
│   │   ├── malex_dataset/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   ├── malimg_dataset/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   ├── malimg_dataset_leaked_backup/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   ├── malimg_dataset_old_backup/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   ├── malimg_dataset_padmatched/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── malimg_dataset_padmatched_v2/
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   ├── benign_images_256x256/
│   ├── benign_images_nataraj/
│   ├── benign_pe_files_test/
│   ├── logs/
│   ├── models/
│   └── venv/
├── run_logs/
└── .vscode/
```

## Project_Resourse Python Files

All current `.py` files in `Project_Resourse/`:

1. `adversarial_train_efficientnet.py`
2. `adversarial_train.py`
3. `build_pad_matched_dataset.py`
4. `check_hash_overlaps.py`
5. `check_raw_png_sizes.py`
6. `check_sizes.py`
7. `collect_benign_pe.py`
8. `config.py`
9. `convert.py`
10. `convert_to_malimg.py`
11. `dataset_loader.py`
14. `evaluate_attacks.py`
15. `fix_dataset_leakage.py`
16. `fix_malimg_crossplit_duplicates.py`
17. `generate_adversarial_test_set.py`
18. `models.py`
19. `rebuild_dataset_1to1.py`
20. `split_benign_dataset.py`
21. `tmp_split_diag_full.py`
22. `train_efficientnet.py`
23. `train_padneutralized_sampler.py`
24. `train.py`
25. `build_malex_split.py`
26. `verify_malex_source.py`
27. `check_malex_hash_overlaps.py`
28. `check_malex_labels.py`
29. `check_malex_shuffle_sanity.py`
30. `verify_benign_collection.py`
31. `verify_cropping_shortcut.py`
32. `verify_image_compatibility.py`
33. `verify_label_shuffling.py`
34. `verify_noise_images.py`
35. `verify_no_pad_subset.py`
36. `verify_padding_shortcut.py`
37. `verify_pad_matched_subset.py`
38. `verify_raw_size_shortcut.py`
39. `verify_visual_duplicates.py`
40. `visualize_clean_dataset.py`

Notes:
- `Project_Resourse` spelling is preserved to match the actual repository folder name.
- Large binary datasets, model weights, and image files are intentionally not expanded in this document.

## MaleX Run Logs (new)

- `run_logs/train_resnet18_malex_stage2.log`
- `run_logs/evaluate_attacks_malex_stage3.log`
- `run_logs/adversarial_train_malex_stage3.log`
