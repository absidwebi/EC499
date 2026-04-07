"""
evaluate_attacks_fixed.py
==========================
Evaluates ANY model against the PRE-GENERATED fixed adversarial test set.

Unlike evaluate_attacks.py (which regenerates attacks on every run),
this script loads the fixed PNGs from adversarial_test_set_malex/ and
evaluates the model once. Results are deterministic and reproducible.

Supports three model variants:
  - "3c2d"             -> clean 3C2D baseline
  - "3c2d_pgd"         -> 3C2D adversarially trained with PGD
  - "3c2d_fgsm"        -> 3C2D adversarially trained with FGSM

Usage:
    # Evaluate clean baseline
    python evaluate_attacks_fixed.py --model 3c2d

    # Evaluate PGD-defended model
    python evaluate_attacks_fixed.py --model 3c2d_pgd

    # Evaluate FGSM-defended model
    python evaluate_attacks_fixed.py --model 3c2d_fgsm

    # Evaluate all three and compare
    python evaluate_attacks_fixed.py --model all

Output:
    run_logs/fixed_adv_eval_{model_tag}_{timestamp}.log
    logs/fixed_adv_eval_{model_tag}.txt
"""

import os
import sys
import argparse
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MALEX_ADV_TEST_SET_DIR_STR,
    MALEX_3C2D_CLEAN_MODEL_PATH_STR,
    MALEX_3C2D_ADV_MODEL_PATH_STR,
    MALEX_3C2D_FGSM_ADV_MODEL_PATH_STR,
    LOGS_DIR,
    PROJECT_ROOT,
)
from models import MaleX3C2D

# ============================================================
# CONFIGURATION
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# run_logs directory (project-level, not Project_Resourse/logs/)
RUN_LOGS_DIR = PROJECT_ROOT.parent / "run_logs"
RUN_LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Model registry
MODEL_REGISTRY = {
    "3c2d": {
        "label": "3C2D Clean Baseline",
        "path": MALEX_3C2D_CLEAN_MODEL_PATH_STR,
        "model_fn": MaleX3C2D,
    },
    "3c2d_pgd": {
        "label": "3C2D Adversarially Trained (PGD)",
        "path": MALEX_3C2D_ADV_MODEL_PATH_STR,
        "model_fn": MaleX3C2D,
    },
    "3c2d_fgsm": {
        "label": "3C2D Adversarially Trained (FGSM)",
        "path": MALEX_3C2D_FGSM_ADV_MODEL_PATH_STR,
        "model_fn": MaleX3C2D,
    },
}

# Fixed adversarial set sub-folders to evaluate
ADV_SUBSETS = [
    "fgsm_eps0.05",
    "pgd_eps0.05_steps40",
]
# ============================================================

TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def _safe_div(num, den):
    return float(num / den) if den else 0.0


def _tpr_at_fpr(labels, probs, target_fpr=0.01):
    """Return max TPR where FPR <= target_fpr."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, probs)
    mask = fpr <= target_fpr
    if not np.any(mask):
        return 0.0
    return float(np.max(tpr[mask]))


def _compute_full_metrics(labels, logits):
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = (preds == labels).mean() * 100.0
    specificity = _safe_div(tn, tn + fp)
    fpr = _safe_div(fp, fp + tn)
    fnr = _safe_div(fn, fn + tp)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float("nan")

    return {
        "accuracy": float(accuracy),
        "confusion_matrix": cm.tolist(),
        "confusion_counts": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "precision_per_class": {
            "benign": _safe_div(tn, tn + fn),
            "malware": _safe_div(tp, tp + fp),
        },
        "recall_per_class": {
            "benign": _safe_div(tn, tn + fp),
            "malware": _safe_div(tp, tp + fn),
        },
        "specificity": float(specificity),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "mcc": float(matthews_corrcoef(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_binary": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
        "auc": float(auc),
        "threshold_metrics": {
            "tpr_at_fpr_1pct": _tpr_at_fpr(labels, probs, target_fpr=0.01),
            "tpr_at_fpr_5pct": _tpr_at_fpr(labels, probs, target_fpr=0.05),
        },
    }


def load_model(model_tag):
    """Load a model by tag. Returns (model, label) or raises."""
    if model_tag not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model tag: {model_tag}. "
            f"Choose from: {list(MODEL_REGISTRY.keys())}"
        )
    cfg = MODEL_REGISTRY[model_tag]
    if not os.path.exists(cfg["path"]):
        raise FileNotFoundError(
            f"Model weights not found: {cfg['path']}\n"
            f"Train the {cfg['label']} model first."
        )
    model = cfg["model_fn"]().to(DEVICE)
    model.load_state_dict(torch.load(cfg["path"], map_location=DEVICE))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Loaded {cfg['label']} ({n_params:,} params)")
    return model, cfg["label"]


@torch.no_grad()
def evaluate_on_subset(model, subset_name):
    """
    Load pre-generated adversarial images and evaluate model recall.

    Returns dict with: recall, evasion_rate, correct, total
    """
    subset_dir = Path(MALEX_ADV_TEST_SET_DIR_STR) / subset_name / "images"
    if not subset_dir.exists():
        raise FileNotFoundError(
            f"Adversarial subset not found: {subset_dir}\n"
            "Run generate_malex_adv_testset.py first."
        )

    png_files = sorted(subset_dir.glob("*.png"))
    if len(png_files) == 0:
        raise ValueError(f"No PNG files found in {subset_dir}")

    correct = 0
    total = len(png_files)

    for fpath in png_files:
        img = TRANSFORM(Image.open(fpath)).unsqueeze(0).to(DEVICE)
        # All images in the set are malware (label=1)
        # We check if model still predicts malware (logit > 0)
        pred = (model(img).squeeze() > 0).long().item()
        if pred == 1:
            correct += 1

    recall = 100.0 * correct / total if total > 0 else 0.0
    evasion = 100.0 - recall
    return {
        "recall": recall,
        "evasion_rate": evasion,
        "correct": correct,
        "total": total,
    }


def evaluate_clean_metrics(model):
    """
    Evaluate model on clean MaleX test set with full metric suite.
    Uses the dataset loader rather than adversarial images.
    """
    from dataset_loader import get_data_loaders
    from config import MALEX_DATASET_DIR_STR

    _, _, test_loader, _ = get_data_loaders(
        data_dir=MALEX_DATASET_DIR_STR, batch_size=64, num_workers=0
    )
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images).squeeze(1)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    return _compute_full_metrics(labels, logits)


def run_evaluation(model_tags, log_lines):
    """Run full evaluation for one or more model tags."""
    results = {}

    for tag in model_tags:
        print(f"\n{'=' * 65}")
        print(f"  Evaluating: {MODEL_REGISTRY[tag]['label']}")
        print(f"{'=' * 65}")

        model, label = load_model(tag)

        # Clean metrics
        print("[*] Measuring clean test metrics...")
        clean_metrics = evaluate_clean_metrics(model)
        print(f"    Clean accuracy: {clean_metrics['accuracy']:.2f}%")
        print(
            f"    TPR@FPR<=1%: {clean_metrics['threshold_metrics']['tpr_at_fpr_1pct']:.4f} "
            f"| MCC: {clean_metrics['mcc']:.4f}"
        )

        # Adversarial accuracy on each subset
        adv_results = {}
        for subset in ADV_SUBSETS:
            print(f"[*] Evaluating on fixed set: {subset}...")
            try:
                r = evaluate_on_subset(model, subset)
                adv_results[subset] = r
                print(
                    f"    Recall={r['recall']:.2f}%  "
                    f"Evasion={r['evasion_rate']:.2f}%  "
                    f"({r['correct']}/{r['total']})"
                )
            except FileNotFoundError as e:
                print(f"    [SKIP] {e}")
                adv_results[subset] = None

        results[tag] = {
            "label": label,
            "clean_metrics": clean_metrics,
            "adv_results": adv_results,
        }

        # Format log lines for this model
        log_lines.append(f"\n{'=' * 65}\n")
        log_lines.append(f"Model: {label}\n")
        log_lines.append(f"{'=' * 65}\n")
        log_lines.append("\nClean Test Metrics (full suite):\n")
        log_lines.append(f"Accuracy            : {clean_metrics['accuracy']:.4f}%\n")
        log_lines.append(
            "Confusion Counts    : "
            f"TN={clean_metrics['confusion_counts']['tn']} "
            f"FP={clean_metrics['confusion_counts']['fp']} "
            f"FN={clean_metrics['confusion_counts']['fn']} "
            f"TP={clean_metrics['confusion_counts']['tp']}\n"
        )
        log_lines.append(
            "Precision (B/M)     : "
            f"{clean_metrics['precision_per_class']['benign']:.4f} / "
            f"{clean_metrics['precision_per_class']['malware']:.4f}\n"
        )
        log_lines.append(
            "Recall (B/M)        : "
            f"{clean_metrics['recall_per_class']['benign']:.4f} / "
            f"{clean_metrics['recall_per_class']['malware']:.4f}\n"
        )
        log_lines.append(f"Specificity (TNR)   : {clean_metrics['specificity']:.4f}\n")
        log_lines.append(f"FPR / FNR           : {clean_metrics['fpr']:.4f} / {clean_metrics['fnr']:.4f}\n")
        log_lines.append(f"Balanced Accuracy   : {clean_metrics['balanced_accuracy']:.4f}\n")
        log_lines.append(f"MCC                 : {clean_metrics['mcc']:.4f}\n")
        log_lines.append(f"F1 Macro            : {clean_metrics['f1_macro']:.4f}\n")
        log_lines.append(f"F1 Malware          : {clean_metrics['f1_binary']:.4f}\n")
        log_lines.append(f"AUC-ROC             : {clean_metrics['auc']:.4f}\n")
        log_lines.append(
            "TPR@FPR<=1% / 5%    : "
            f"{clean_metrics['threshold_metrics']['tpr_at_fpr_1pct']:.4f} / "
            f"{clean_metrics['threshold_metrics']['tpr_at_fpr_5pct']:.4f}\n"
        )
        log_lines.append("\nAdversarial Evaluation (Fixed Test Set):\n")
        log_lines.append(f"{'Attack':<35} {'Recall':>8} {'Evasion':>9} {'Correct':>9} {'Total':>7}\n")
        log_lines.append("-" * 72 + "\n")
        for subset, r in adv_results.items():
            if r is None:
                log_lines.append(f"{subset:<35} {'SKIP':>8}\n")
            else:
                log_lines.append(
                    f"{subset:<35} {r['recall']:>7.2f}% {r['evasion_rate']:>8.2f}%"
                    f" {r['correct']:>9} {r['total']:>7}\n"
                )

    return results


def print_comparison_table(results):
    """Print a side-by-side comparison table if multiple models evaluated."""
    if len(results) < 2:
        return

    print("\n" + "=" * 80)
    print("  COMPARISON TABLE - Clean vs Defended Models")
    print("=" * 80)

    models = list(results.keys())
    header = f"{'Attack':<35}"
    for tag in models:
        label = results[tag]["label"].split("(")[-1].rstrip(")")[:12]
        header += f" {label:>14}"
    print(header)
    print("-" * 80)

    # Clean accuracy row
    row = f"{'Clean Accuracy':<35}"
    for tag in models:
        row += f" {results[tag]['clean_metrics']['accuracy']:>13.2f}%"
    print(row)

    # Adversarial rows
    for subset in ADV_SUBSETS:
        row_recall = f"{subset + ' (Recall)':<35}"
        row_evasion = f"{subset + ' (Evasion)':<35}"
        for tag in models:
            r = results[tag]["adv_results"].get(subset)
            if r is None:
                row_recall += f" {'SKIP':>14}"
                row_evasion += f" {'SKIP':>14}"
            else:
                row_recall += f" {r['recall']:>13.2f}%"
                row_evasion += f" {r['evasion_rate']:>13.2f}%"
        print(row_recall)
        print(row_evasion)

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models on fixed MaleX adversarial test set"
    )
    parser.add_argument(
        "--model",
        default="3c2d",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        help="Model variant to evaluate (default: 3c2d)",
    )
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_tags = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    tag_str = args.model

    print("=" * 65)
    print("  EC499 - Fixed Adversarial Test Set Evaluation")
    print(f"  Model(s): {', '.join(model_tags)}")
    print(f"  Device  : {DEVICE}")
    print(f"  Adv set : {MALEX_ADV_TEST_SET_DIR_STR}")
    print("=" * 65)

    log_lines = [
        "EC499 - Fixed Adversarial Test Set Evaluation\n",
        f"Timestamp : {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"Model(s)  : {', '.join(model_tags)}\n",
        f"Device    : {DEVICE}\n",
        f"Adv set   : {MALEX_ADV_TEST_SET_DIR_STR}\n",
    ]

    results = run_evaluation(model_tags, log_lines)

    # Print comparison table
    print_comparison_table(results)

    # Write logs
    run_log_path = RUN_LOGS_DIR / f"fixed_adv_eval_{tag_str}_{timestamp}.log"
    perm_log_path = LOGS_DIR / f"fixed_adv_eval_{tag_str}.txt"

    log_content = "".join(log_lines)
    with open(run_log_path, "w") as f:
        f.write(log_content)
    with open(perm_log_path, "w") as f:
        f.write(log_content)

    print(f"\n[*] Run log  : {run_log_path}")
    print(f"[*] Perm log : {perm_log_path}")
    print("[DONE] Evaluation complete.")


if __name__ == "__main__":
    main()
