# EC499 Folder Structure Overview (Updated)

Last updated: 2026-03-28

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
12. `evaluate_attacks.py`
13. `fix_dataset_leakage.py`
14. `fix_malimg_crossplit_duplicates.py`
15. `generate_adversarial_test_set.py`
16. `models.py`
17. `rebuild_dataset_1to1.py`
18. `split_benign_dataset.py`
19. `tmp_split_diag_full.py`
20. `train_efficientnet.py`
21. `train_padneutralized_sampler.py`
22. `train.py`
23. `verify_benign_collection.py`
24. `verify_cropping_shortcut.py`
25. `verify_image_compatibility.py`
26. `verify_label_shuffling.py`
27. `verify_noise_images.py`
28. `verify_no_pad_subset.py`
29. `verify_padding_shortcut.py`
30. `verify_pad_matched_subset.py`
31. `verify_raw_size_shortcut.py`
32. `verify_visual_duplicates.py`
33. `visualize_clean_dataset.py`

Notes:
- `Project_Resourse` spelling is preserved to match the actual repository folder name.
- Large binary datasets, model weights, and image files are intentionally not expanded in this document.
