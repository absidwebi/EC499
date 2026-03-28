import argparse
import os
import random
import warnings
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import LOGS_DIR, MALIMG_ARCHIVE_DIR


SEED = 42


def _iter_png_files(class_dir):
    for f in os.listdir(class_dir):
        if f.lower().endswith(".png"):
            yield os.path.join(class_dir, f)


def _load_split_features(split_dir, max_per_class=None):
    # Features: width, height, aspect, area
    X = []
    y = []

    class_names = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    for class_name in sorted(class_names):
        label = 0 if class_name == "benign" else 1
        class_dir = os.path.join(split_dir, class_name)
        paths = list(_iter_png_files(class_dir))
        if max_per_class is not None and len(paths) > max_per_class:
            paths = random.sample(paths, max_per_class)

        for p in paths:
            try:
                with Image.open(p) as im:
                    w, h = im.size
            except Exception:
                continue
            aspect = w / max(1.0, float(h))
            area = float(w) * float(h)
            X.append([float(w), float(h), float(aspect), float(area)])
            y.append(label)

    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)


def _summarize(name, X, y):
    for cls, cname in [(0, "benign"), (1, "malware")]:
        Xi = X[y == cls]
        if Xi.size == 0:
            continue
        w_mean = Xi[:, 0].mean()
        h_mean = Xi[:, 1].mean()
        print(f"[{name}] {cname}: n={Xi.shape[0]} width_mean={w_mean:.2f} height_mean={h_mean:.2f}")


def main(max_train_per_class=5000, max_val_per_class=2000, max_test_per_class=2000, make_plots=True):
    random.seed(SEED)
    np.random.seed(SEED)

    warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

    train_dir = str(MALIMG_ARCHIVE_DIR / "train")
    val_dir = str(MALIMG_ARCHIVE_DIR / "val")
    test_dir = str(MALIMG_ARCHIVE_DIR / "test")

    X_train, y_train = _load_split_features(train_dir, max_per_class=max_train_per_class)
    X_val, y_val = _load_split_features(val_dir, max_per_class=max_val_per_class)
    X_test, y_test = _load_split_features(test_dir, max_per_class=max_test_per_class)

    print("== Raw PNG size shift ==")
    _summarize("train", X_train, y_train)
    _summarize("val", X_val, y_val)
    _summarize("test", X_test, y_test)

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=SEED,
                    max_iter=500,
                ),
            ),
        ]
    )
    clf.fit(X_train, y_train)

    def eval_split(name, X, y):
        prob = clf.predict_proba(X)[:, 1]
        pred = (prob >= 0.5).astype(np.int64)
        auc = roc_auc_score(y, prob)
        acc = accuracy_score(y, pred)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        print(f"\n== Size-only LR on {name} ==")
        print(f"roc_auc={auc:.4f} accuracy={acc:.4f}")
        print("confusion_matrix rows=true [benign,malware], cols=pred [benign,malware]")
        print(cm)
        return prob, auc

    prob_val, auc_val = eval_split("val", X_val, y_val)
    prob_test, auc_test = eval_split("test", X_test, y_test)

    if not make_plots:
        return

    out_dir = Path(LOGS_DIR)
    out_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(6.5, 5))
    for name, y, prob, auc in [
        ("val", y_val, prob_val, auc_val),
        ("test", y_test, prob_test, auc_test),
    ]:
        fpr, tpr, _ = roc_curve(y, prob)
        plt.plot(fpr, tpr, label=f"{name} AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Raw PNG Size-only Logistic Regression")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_roc = out_dir / "raw_size_only_lr_roc.png"
    plt.savefig(out_roc, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"roc_plot_saved={out_roc}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Verify raw PNG size shortcut")
    p.add_argument("--max-train-per-class", type=int, default=5000)
    p.add_argument("--max-val-per-class", type=int, default=2000)
    p.add_argument("--max-test-per-class", type=int, default=2000)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()
    main(
        max_train_per_class=args.max_train_per_class,
        max_val_per_class=args.max_val_per_class,
        max_test_per_class=args.max_test_per_class,
        make_plots=(not args.no_plots),
    )
