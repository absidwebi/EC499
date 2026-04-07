"""
Stage 3 — Part 1: Adversarial Attack Evaluation
================================================
Loads the trained 'vulnerable' baseline CNN and evaluates its accuracy
under FGSM and PGD white-box adversarial attacks.

Purpose:
    Demonstrate the catastrophic drop in accuracy that occurs when the
    undefended model is subjected to gradient-based evasion attacks.
    This motivates the adversarial training defense in adversarial_train.py.

NOTE on attack implementation:
    torchattacks internally uses CrossEntropyLoss, which is incompatible
    with our single-logit BCEWithLogitsLoss binary model. We therefore
    implement FGSM and PGD manually using BCEWithLogitsLoss directly.

Output:
    A results table comparing Clean Accuracy vs Robust Accuracy across
    multiple attack types and epsilon perturbation strengths.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)

from config import (
    MALEX_DATASET_DIR_STR,
    MALEX_3C2D_CLEAN_MODEL_PATH_STR,
    RESNET_MALEX_CLEAN_MODEL_PATH_STR,
    RESNET_MALEX_PRETRAINED_CLEAN_PATH_STR,
    LOGS_DIR,
)
from dataset_loader import get_data_loaders
from models import MaleX3C2D, get_resnet18_grayscale, get_resnet18_pretrained_grayscale

# === CONFIGURATION ===
BATCH_SIZE    = 32
FGSM_EPSILONS = [0.01, 0.02, 0.05, 0.1]
PGD_CONFIGS   = [
    {"eps": 0.01, "alpha": 0.0025, "steps": 10},
    {"eps": 0.02, "alpha": 0.005,  "steps": 20},
    {"eps": 0.05, "alpha": 0.01,   "steps": 40},
]
criterion = nn.BCEWithLogitsLoss()

# Model options: "3c2d", "resnet", "resnet_pretrained"
MODEL_VARIANT = "3c2d"
# =====================


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


def get_model_and_path(model_variant):
    variant = model_variant.lower().strip()
    if variant == "3c2d":
        return "MaleX3C2D", MaleX3C2D(), MALEX_3C2D_CLEAN_MODEL_PATH_STR
    if variant == "resnet":
        return "ResNet-18", get_resnet18_grayscale(), RESNET_MALEX_CLEAN_MODEL_PATH_STR
    if variant == "resnet_pretrained":
        return (
            "ResNet-18 Pretrained",
            get_resnet18_pretrained_grayscale(),
            RESNET_MALEX_PRETRAINED_CLEAN_PATH_STR,
        )
    raise ValueError(
        f"Unsupported MODEL_VARIANT='{model_variant}'. "
        "Use one of: '3c2d', 'resnet', 'resnet_pretrained'."
    )


def fgsm_attack(model, images, labels, eps):
    """Fast Gradient Sign Method (Goodfellow et al., 2014)."""
    images = images.clone().detach().requires_grad_(True)
    labels_f = labels.float().unsqueeze(1)

    outputs = model(images)
    loss = criterion(outputs, labels_f)
    model.zero_grad()
    loss.backward()

    perturbation = eps * images.grad.sign()
    adv_images = torch.clamp(images + perturbation, -1.0, 1.0).detach()
    return adv_images


def pgd_attack(model, images, labels, eps, alpha, steps):
    """Projected Gradient Descent (Madry et al., 2018)."""
    labels_f = labels.float().unsqueeze(1)
    # Start from a random point inside the epsilon ball
    adv_images = images.clone().detach()
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
    adv_images = torch.clamp(adv_images, -1.0, 1.0).detach()

    for _ in range(steps):
        adv_images.requires_grad_(True)
        outputs = model(adv_images)
        loss = criterion(outputs, labels_f)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            adv_images = adv_images + alpha * adv_images.grad.sign()
            # Project back into the epsilon ball around the original image
            delta = torch.clamp(adv_images - images, -eps, eps)
            adv_images = torch.clamp(images + delta, -1.0, 1.0).detach()

    return adv_images



def evaluate(model, loader, device, attack_fn=None, attack_kwargs=None):
    """
    Evaluate model on a DataLoader with optional adversarial attack.
    attack_fn: callable(model, images, labels, **kwargs) -> adv_images
    Returns a dict of full confusion-style metrics.
    """
    model.eval()
    all_logits = []
    all_labels = []
    if attack_kwargs is None:
        attack_kwargs = {}

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if attack_fn is not None:
            images = attack_fn(model, images, labels, **attack_kwargs)

        with torch.no_grad():
            outputs = model(images).squeeze(1)  # Raw logits [B]
            all_logits.append(outputs.cpu())
            all_labels.append(labels.cpu())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    return _compute_full_metrics(labels, logits)


def main():
    print("=" * 60)
    print("  Stage 3 — Part 1: Adversarial Attack Evaluation")
    print("=" * 60)

    # 1. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")
    print(f"[*] Selected model variant: {MODEL_VARIANT}")

    # 2. Load Data (test set only needed for evaluation)
    print("[*] Loading test dataset...")
    _, _, test_loader, _ = get_data_loaders(
        data_dir=MALEX_DATASET_DIR_STR,
        batch_size=BATCH_SIZE,
        num_workers=0
    )

    # 3. Load the Vulnerable Baseline Model
    model_name, model, model_path = get_model_and_path(MODEL_VARIANT)
    print(f"[*] Loading {model_name} from: {model_path}")
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. Baseline Clean Accuracy
    print("\n[*] Measuring Clean Accuracy (no attack)...")
    clean_metrics = evaluate(model, test_loader, device)
    clean_acc = clean_metrics["accuracy"]
    print(f"    ✅ Clean Accuracy: {clean_acc:.2f}%")
    print(
        f"    Clean TPR@FPR<=1%: {clean_metrics['threshold_metrics']['tpr_at_fpr_1pct']:.4f} "
        f"| MCC: {clean_metrics['mcc']:.4f}"
    )

    results = [{
        "name": "Clean (No Attack)",
        "epsilon": "—",
        "metrics": clean_metrics,
    }]

    # 5. FGSM Attacks
    print("\n[*] Running FGSM Attacks...")
    for eps in FGSM_EPSILONS:
        metrics = evaluate(model, test_loader, device,
                           attack_fn=fgsm_attack, attack_kwargs={"eps": eps})
        acc = metrics["accuracy"]
        tag = f"FGSM (e={eps})"
        print(f"    {'OK' if acc > 90 else 'WARN'} {tag}: {acc:.2f}%  (dropped {clean_acc - acc:.2f}%)")
        print(f"         TPR@FPR<=1%={metrics['threshold_metrics']['tpr_at_fpr_1pct']:.4f}, MCC={metrics['mcc']:.4f}")
        results.append({
            "name": tag,
            "epsilon": f"e={eps}",
            "metrics": metrics,
        })

    # 6. PGD Attacks
    print("\n[*] Running PGD Attacks...")
    for cfg in PGD_CONFIGS:
        metrics = evaluate(model, test_loader, device,
                           attack_fn=pgd_attack,
                           attack_kwargs={"eps": cfg["eps"], "alpha": cfg["alpha"], "steps": cfg["steps"]})
        acc = metrics["accuracy"]
        tag = f"PGD  (e={cfg['eps']}, steps={cfg['steps']})"
        print(f"    {'FAIL' if acc < 50 else 'WARN'} {tag}: {acc:.2f}%  (dropped {clean_acc - acc:.2f}%)")
        print(f"         TPR@FPR<=1%={metrics['threshold_metrics']['tpr_at_fpr_1pct']:.4f}, MCC={metrics['mcc']:.4f}")
        results.append({
            "name": tag,
            "epsilon": f"e={cfg['eps']}",
            "metrics": metrics,
        })

    # 7. Print Summary Table
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Attack':<35} {'Epsilon':<12} {'Accuracy':>10}")
    print("-" * 60)
    for item in results:
        name = item["name"]
        epsilon = item["epsilon"]
        acc = item["metrics"]["accuracy"]
        status = "✅" if acc > 90 else ("⚠️ " if acc > 50 else "🔴")
        print(f"{status} {name:<33} {epsilon:<12} {acc:>9.2f}%")
    print("=" * 60)

    # 8. Save results to log file
    model_tag = MODEL_VARIANT.lower().replace(" ", "_")
    log_path = LOGS_DIR / f"attack_evaluation_results_{model_tag}.txt"
    with open(log_path, "w") as f:
        f.write(f"Stage 3 — Adversarial Attack Evaluation Results ({model_name})\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Attack':<35} {'Epsilon':<12} {'Accuracy':>10}\n")
        f.write("-" * 60 + "\n")
        for item in results:
            f.write(
                f"{item['name']:<35} {item['epsilon']:<12} "
                f"{item['metrics']['accuracy']:>9.2f}%\n"
            )
        f.write("\n" + "=" * 60 + "\n")
        f.write("Detailed Metrics Per Setting\n")
        f.write("=" * 60 + "\n")
        for item in results:
            m = item["metrics"]
            f.write(f"\n[{item['name']}] ({item['epsilon']})\n")
            f.write(
                "Confusion Counts: "
                f"TN={m['confusion_counts']['tn']} FP={m['confusion_counts']['fp']} "
                f"FN={m['confusion_counts']['fn']} TP={m['confusion_counts']['tp']}\n"
            )
            f.write(
                "Precision (B/M): "
                f"{m['precision_per_class']['benign']:.4f} / "
                f"{m['precision_per_class']['malware']:.4f}\n"
            )
            f.write(
                "Recall (B/M): "
                f"{m['recall_per_class']['benign']:.4f} / "
                f"{m['recall_per_class']['malware']:.4f}\n"
            )
            f.write(f"Specificity (TNR): {m['specificity']:.4f}\n")
            f.write(f"FPR / FNR: {m['fpr']:.4f} / {m['fnr']:.4f}\n")
            f.write(f"Balanced Accuracy: {m['balanced_accuracy']:.4f}\n")
            f.write(f"MCC: {m['mcc']:.4f}\n")
            f.write(f"F1 Macro / Malware: {m['f1_macro']:.4f} / {m['f1_binary']:.4f}\n")
            f.write(f"AUC-ROC: {m['auc']:.4f}\n")
            f.write(
                "TPR@FPR<=1% / 5%: "
                f"{m['threshold_metrics']['tpr_at_fpr_1pct']:.4f} / "
                f"{m['threshold_metrics']['tpr_at_fpr_5pct']:.4f}\n"
            )
    print(f"\n[*] Results saved to: {log_path}")
    print("\nConclusion: The clean model is highly vulnerable to adversarial perturbations.")
    print("   → Proceed to adversarial_train.py to train a robust defended model.")

if __name__ == "__main__":
    main()
