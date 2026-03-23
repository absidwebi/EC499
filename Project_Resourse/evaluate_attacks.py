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

from config import MALIMG_ARCHIVE_DIR_STR, RESNET_CLEAN_MODEL_PATH_STR, LOGS_DIR
from dataset_loader import get_data_loaders
from models import get_resnet18_grayscale

# === CONFIGURATION ===
BATCH_SIZE    = 32
FGSM_EPSILONS = [0.01, 0.02, 0.05, 0.1]
PGD_CONFIGS   = [
    {"eps": 0.01, "alpha": 0.0025, "steps": 10},
    {"eps": 0.02, "alpha": 0.005,  "steps": 20},
    {"eps": 0.05, "alpha": 0.01,   "steps": 40},
]
criterion = nn.BCEWithLogitsLoss()
MODEL_PATH = RESNET_CLEAN_MODEL_PATH_STR
# =====================


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
    Evaluate model accuracy on a DataLoader, optionally under an adversarial attack.
    attack_fn: callable(model, images, labels, **kwargs) -> adv_images
    Returns accuracy as a float percentage.
    """
    model.eval()
    correct = 0
    total   = 0
    if attack_kwargs is None:
        attack_kwargs = {}

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if attack_fn is not None:
            images = attack_fn(model, images, labels, **attack_kwargs)

        with torch.no_grad():
            outputs   = model(images)                      # Raw logits [B, 1]
            predicted = (outputs > 0).float().squeeze(1)  # Threshold at 0
            correct  += (predicted == labels.float()).sum().item()
            total    += labels.size(0)

    return (correct / total) * 100.0


def main():
    print("=" * 60)
    print("  Stage 3 — Part 1: Adversarial Attack Evaluation")
    print("=" * 60)

    # 1. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    # 2. Load Data (test set only needed for evaluation)
    print("[*] Loading test dataset...")
    _, _, test_loader, _ = get_data_loaders(
        data_dir=MALIMG_ARCHIVE_DIR_STR,
        batch_size=BATCH_SIZE,
        num_workers=0
    )

    # 3. Load the Vulnerable Baseline Model
    print(f"[*] Loading model from: {MODEL_PATH}")
    model = get_resnet18_grayscale().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 4. Baseline Clean Accuracy
    print("\n[*] Measuring Clean Accuracy (no attack)...")
    clean_acc = evaluate(model, test_loader, device)
    print(f"    ✅ Clean Accuracy: {clean_acc:.2f}%")

    results = [("Clean (No Attack)", "—", clean_acc)]

    # 5. FGSM Attacks
    print("\n[*] Running FGSM Attacks...")
    for eps in FGSM_EPSILONS:
        acc = evaluate(model, test_loader, device,
                       attack_fn=fgsm_attack, attack_kwargs={"eps": eps})
        tag = f"FGSM (e={eps})"
        print(f"    {'OK' if acc > 90 else 'WARN'} {tag}: {acc:.2f}%  (dropped {clean_acc - acc:.2f}%)")
        results.append((tag, f"e={eps}", acc))

    # 6. PGD Attacks
    print("\n[*] Running PGD Attacks...")
    for cfg in PGD_CONFIGS:
        acc = evaluate(model, test_loader, device,
                       attack_fn=pgd_attack,
                       attack_kwargs={"eps": cfg["eps"], "alpha": cfg["alpha"], "steps": cfg["steps"]})
        tag = f"PGD  (e={cfg['eps']}, steps={cfg['steps']})"
        print(f"    {'FAIL' if acc < 50 else 'WARN'} {tag}: {acc:.2f}%  (dropped {clean_acc - acc:.2f}%)")
        results.append((tag, f"e={cfg['eps']}", acc))

    # 7. Print Summary Table
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Attack':<35} {'Epsilon':<12} {'Accuracy':>10}")
    print("-" * 60)
    for name, epsilon, acc in results:
        status = "✅" if acc > 90 else ("⚠️ " if acc > 50 else "🔴")
        print(f"{status} {name:<33} {epsilon:<12} {acc:>9.2f}%")
    print("=" * 60)

    # 8. Save results to log file
    log_path = LOGS_DIR / "attack_evaluation_results.txt"
    with open(log_path, "w") as f:
        f.write("Stage 3 — Adversarial Attack Evaluation Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Attack':<35} {'Epsilon':<12} {'Accuracy':>10}\n")
        f.write("-" * 60 + "\n")
        for name, epsilon, acc in results:
            f.write(f"{name:<35} {epsilon:<12} {acc:>9.2f}%\n")
    print(f"\n[*] Results saved to: {log_path}")
    print("\nConclusion: The clean model is highly vulnerable to adversarial perturbations.")
    print("   → Proceed to adversarial_train.py to train a robust defended model.")

if __name__ == "__main__":
    main()
