"""
evaluate.py
-----------
Post-training evaluation:
  - ROC / AUC curves (one-vs-rest)
  - Precision-Recall curves
  - Confusion matrix heatmap
  - Feature importance bar chart
  - Actual vs Predicted scatter (regression)
  - Cross-validation learning curve
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import learning_curve

from preprocessing import run_preprocessing

MODEL_DIR  = Path("models")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

STYLE = {
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
}
plt.rcParams.update(STYLE)

GRADE_COLORS = {
    "Fail": "#E24B4A",
    "Average": "#BA7517",
    "Pass": "#378ADD",
    "Distinction": "#1D9E75",
}


def load_all():
    clf        = joblib.load(MODEL_DIR / "best_classifier.pkl")
    reg        = joblib.load(MODEL_DIR / "best_regressor.pkl")
    le         = joblib.load(MODEL_DIR / "label_encoder.pkl")
    feat_names = joblib.load(MODEL_DIR / "feature_names.pkl")
    return clf, reg, le, feat_names


# ── 1. Confusion matrix ───────────────────────────────────────────────────────

def plot_confusion_matrix(clf, X_test, y_test, label_names):
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_estimator(
        clf, X_test, y_test,
        display_labels=label_names,
        cmap="Blues", ax=ax,
    )
    ax.set_title("Confusion Matrix — Best Classifier", fontsize=13, pad=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print("  [saved] confusion_matrix.png")


# ── 2. ROC curves (one-vs-rest) ───────────────────────────────────────────────

def plot_roc_curves(clf, X_test, y_test, label_names):
    y_bin = label_binarize(y_test, classes=range(len(label_names)))
    if not hasattr(clf, "predict_proba"):
        print("  [skip] ROC — model has no predict_proba")
        return

    y_score = clf.predict_proba(X_test)

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, name in enumerate(label_names):
        RocCurveDisplay.from_predictions(
            y_bin[:, i], y_score[:, i],
            name=name,
            color=list(GRADE_COLORS.values())[i],
            ax=ax,
        )
    auc = roc_auc_score(y_bin, y_score, multi_class="ovr", average="macro")
    ax.set_title(f"ROC Curves (one-vs-rest)  —  macro AUC = {auc:.3f}", fontsize=12)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)
    print("  [saved] roc_curves.png")


# ── 3. Feature importance ─────────────────────────────────────────────────────

def plot_feature_importance(clf, feat_names):
    inner = clf.named_steps["model"]
    if not hasattr(inner, "feature_importances_"):
        print("  [skip] Feature importance — not available for this estimator")
        return

    imps = inner.feature_importances_
    idx  = np.argsort(imps)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#378ADD" if i < 3 else "#B5D4F4" for i in range(len(imps))]
    ax.barh([feat_names[i] for i in idx[::-1]], imps[idx[::-1]], color=colors[::-1])
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance — Best Classifier", fontsize=12, pad=10)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)
    print("  [saved] feature_importance.png")


# ── 4. Actual vs Predicted (regression) ──────────────────────────────────────

def plot_actual_vs_predicted(reg, X_test, y_test):
    y_pred = reg.predict(X_test)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_pred, alpha=0.35, s=18, color="#378ADD")
    lim = (0, 100)
    ax.plot(lim, lim, "r--", lw=1.2, label="Perfect prediction")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual Score"); ax.set_ylabel("Predicted Score")
    ax.set_title("Actual vs Predicted Score (Regression)", fontsize=12, pad=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)
    print("  [saved] actual_vs_predicted.png")


# ── 5. Learning curve ─────────────────────────────────────────────────────────

def plot_learning_curve(clf, X_train, y_train):
    sizes, train_scores, val_scores = learning_curve(
        clf, X_train, y_train,
        cv=5, scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=-1,
    )
    train_mean = train_scores.mean(axis=1)
    val_mean   = val_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sizes, train_mean, "o-", color="#378ADD", label="Training score")
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.12, color="#378ADD")
    ax.plot(sizes, val_mean, "s-", color="#1D9E75", label="CV score")
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.12, color="#1D9E75")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning Curve — Best Classifier", fontsize=12, pad=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "learning_curve.png", dpi=150)
    plt.close(fig)
    print("  [saved] learning_curve.png")


# ── 6. Model comparison bar chart ────────────────────────────────────────────

def plot_model_comparison():
    path = MODEL_DIR / "classifier_results.csv"
    if not path.exists():
        print("  [skip] Model comparison — classifier_results.csv not found")
        return

    df = pd.read_csv(path).sort_values("Accuracy")
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#B5D4F4"] * len(df)
    colors[-1] = "#378ADD"  # highlight best
    ax.barh(df["Model"], df["Accuracy"], color=colors)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["Accuracy"] + 0.3, i, f"{row['Accuracy']}%", va="center", fontsize=10)
    ax.set_xlabel("Test Accuracy (%)")
    ax.set_title("Model Comparison — Classification Accuracy", fontsize=12, pad=10)
    ax.set_xlim(0, 105)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)
    print("  [saved] model_comparison.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_evaluation(csv_path: str = "student_data.csv"):
    print("=" * 50)
    print("  Generating evaluation plots …")
    print("=" * 50)

    clf, reg, le, feat_names = load_all()
    splits, _, _ = run_preprocessing(csv_path)
    label_names  = list(le.classes_)

    X_test  = splits["X_test"]
    y_test  = splits["y_clf_test"]
    X_train = splits["X_train"]
    y_train = splits["y_clf_train"]
    yr_test = splits["y_reg_test"]

    plot_confusion_matrix(clf, X_test, y_test, label_names)
    plot_roc_curves(clf, X_test, y_test, label_names)
    plot_feature_importance(clf, feat_names)
    plot_actual_vs_predicted(reg, X_test, yr_test)
    plot_learning_curve(clf, X_train, y_train)
    plot_model_comparison()

    print(f"\n  All plots saved to → {OUTPUT_DIR}/")


if __name__ == "__main__":
    run_evaluation("student_data.csv")
