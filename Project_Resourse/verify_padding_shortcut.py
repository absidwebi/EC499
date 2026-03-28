import argparse
import random
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import LOGS_DIR, MALIMG_ARCHIVE_DIR_STR
from dataset_loader import get_data_loaders


SEED = 42
BATCH_SIZE = 256
NUM_WORKERS = 0


def _extract_padding_features(loader, max_images=None):
    """Return (X, y) where X uses only padding-derived features.

    Features (per image):
      - pad_rows_all_neg1
      - pad_cols_all_neg1
      - frac_pixels_neg1
    """
    feats = []
    labels = []
    seen = 0

    for bi, (imgs, lbls) in enumerate(loader):
        # imgs: [B,1,256,256] normalized to [-1,1]
        x = imgs.squeeze(1)
        is_neg1 = (x == -1.0)

        pad_rows = is_neg1.all(dim=2).sum(dim=1).to(torch.int64)
        pad_cols = is_neg1.all(dim=1).sum(dim=1).to(torch.int64)
        frac_neg1 = is_neg1.float().mean(dim=(1, 2))

        batch_feats = torch.stack(
            [pad_rows.float(), pad_cols.float(), frac_neg1], dim=1
        ).cpu().numpy()
        feats.append(batch_feats)
        labels.append(lbls.cpu().numpy().astype(np.int64))

        seen += imgs.size(0)
        if bi % 25 == 0:
            print(f"  batches={bi} images={seen}")
        if max_images is not None and seen >= max_images:
            break

    X = np.concatenate(feats, axis=0)
    y = np.concatenate(labels, axis=0)
    if max_images is not None:
        X = X[:max_images]
        y = y[:max_images]
    return X, y


def _summarize_features(name, X, y):
    def s(arr):
        arr = np.asarray(arr)
        return float(arr.mean()), float(np.quantile(arr, 0.95))

    for cls, cname in [(0, "benign"), (1, "malware")]:
        Xi = X[y == cls]
        r_mean, r_p95 = s(Xi[:, 0])
        c_mean, c_p95 = s(Xi[:, 1])
        f_mean, f_p95 = s(Xi[:, 2])
        print(f"[{name}] {cname}: n={Xi.shape[0]}")
        print(f"  pad_rows_all_-1 mean={r_mean:.2f} p95={r_p95:.2f}")
        print(f"  pad_cols_all_-1 mean={c_mean:.2f} p95={c_p95:.2f}")
        print(f"  frac_-1_pixels  mean={f_mean:.4f} p95={f_p95:.4f}")


def main(max_train=None, max_val=None, max_test=None, make_plots=True):
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    train_loader, val_loader, test_loader, _ = get_data_loaders(
        MALIMG_ARCHIVE_DIR_STR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    X_train, y_train = _extract_padding_features(train_loader, max_images=max_train)
    X_val, y_val = _extract_padding_features(val_loader, max_images=max_val)
    X_test, y_test = _extract_padding_features(test_loader, max_images=max_test)

    print("== Padding feature shift (normalized tensors) ==")
    _summarize_features("train", X_train, y_train)
    _summarize_features("val", X_val, y_val)
    _summarize_features("test", X_test, y_test)

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
        print(f"\n== Padding-only LR on {name} ==")
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

    # ROC curves
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
    plt.title("Padding-only Logistic Regression")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_roc = out_dir / "padding_only_lr_roc.png"
    plt.savefig(out_roc, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"roc_plot_saved={out_roc}")

    # Feature distributions (pad_cols)
    plt.figure(figsize=(8, 4.5))
    for cls, cname, color in [(0, "benign", "tab:blue"), (1, "malware", "tab:orange")]:
        plt.hist(
            X_train[y_train == cls, 1],
            bins=50,
            alpha=0.5,
            density=True,
            label=f"train {cname}",
            color=color,
        )
    plt.xlabel("pad_cols_all_-1")
    plt.ylabel("density")
    plt.title("Train Distribution: pad_cols_all_-1")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_hist = out_dir / "padding_feature_pad_cols_hist_train.png"
    plt.savefig(out_hist, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"feature_hist_saved={out_hist}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Verify padding-only shortcut")
    p.add_argument("--max-train", type=int, default=5000)
    p.add_argument("--max-val", type=int, default=2000)
    p.add_argument("--max-test", type=int, default=2000)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()
    main(
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
        make_plots=(not args.no_plots),
    )
