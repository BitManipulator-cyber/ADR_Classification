# ============================================================
# CELL 7a — Visualizations for ML Baselines
# Self-contained: retrains nb, lr, svm from saved cleaned data
# ============================================================

import os, gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve,
    average_precision_score, f1_score,
    classification_report
)
from sklearn.preprocessing import label_binarize

os.makedirs("outputs/ml", exist_ok=True)

CLASSES     = label_encoder.classes_
COLORS      = ["#378ADD", "#1D9E75", "#D85A30"]
MODEL_NAMES = ["Naïve Bayes", "Logistic Regression", "SVM (LinearSVC)"]

# ── Retrain TF-IDF + all 3 models from saved splits ──────────
print("Rebuilding TF-IDF and retraining ML models...")
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=8000, sublinear_tf=True)
X_tr_tfidf = tfidf.fit_transform(X_train_clean)
X_te_tfidf = tfidf.transform(X_test_clean)

nb = MultinomialNB()
nb.fit(X_tr_tfidf, y_train)
print("  ✓ Naïve Bayes trained")

lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                        random_state=SEED, solver="saga", n_jobs=-1)
lr.fit(X_tr_tfidf, y_train)
print("  ✓ Logistic Regression trained")

svm_base = LinearSVC(class_weight="balanced", random_state=SEED, max_iter=2000)
svm = CalibratedClassifierCV(svm_base, cv=3)
svm.fit(X_tr_tfidf, y_train)
print("  ✓ SVM trained")

MODELS = [nb, lr, svm]

# ── Helper: predictions + probabilities ──────────────────────
def get_preds_probs(model, X_te):
    y_pred = model.predict(X_te)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_te)
    else:
        df = model.decision_function(X_te)
        df = df - df.max(axis=1, keepdims=True)
        e  = np.exp(df)
        y_prob = e / e.sum(axis=1, keepdims=True)
    return y_pred, y_prob

# ── Plot 1: Confusion Matrices ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, model, name in zip(axes, MODELS, MODEL_NAMES):
    y_pred, _ = get_preds_probs(model, X_te_tfidf)
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name}\nConfusion Matrix", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.suptitle("ML Baselines — Confusion Matrices",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/ml/ml_confusion_matrices.png",
            dpi=130, bbox_inches="tight")
plt.close()
print("✓ ml_confusion_matrices.png saved")

# ── Plot 2: Per-class F1 grouped bar chart ────────────────────
per_class_f1 = {}
for model, name in zip(MODELS, MODEL_NAMES):
    y_pred, _ = get_preds_probs(model, X_te_tfidf)
    scores = f1_score(y_test, y_pred, average=None,
                      labels=[0, 1, 2], zero_division=0)
    per_class_f1[name] = scores

x     = np.arange(len(CLASSES))
width = 0.25
fig, ax = plt.subplots(figsize=(10, 5))
for i, (name, scores) in enumerate(per_class_f1.items()):
    bars = ax.bar(x + i * width, scores, width,
                  label=name, color=COLORS[i], edgecolor="white")
    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x + width)
ax.set_xticklabels(CLASSES)
ax.set_ylim(0, 1.1)
ax.set_ylabel("F1 Score")
ax.set_title("Per-class F1 Score — ML Baselines",
             fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/ml/ml_per_class_f1.png",
            dpi=130, bbox_inches="tight")
plt.close()
print("✓ ml_per_class_f1.png saved")

# ── Plot 3: Overall metrics bar chart ─────────────────────────
ml_results = []
for model, name in zip(MODELS, MODEL_NAMES):
    y_pred, _ = get_preds_probs(model, X_te_tfidf)
    ml_results.append({
        "Model"    : name,
        "Accuracy" : round(float(np.mean(y_pred == y_test)), 4),
        "Precision": round(float(
            __import__("sklearn.metrics", fromlist=["precision_score"])
            .precision_score(y_test, y_pred, average="weighted",
                             zero_division=0)), 4),
        "Recall"   : round(float(
            __import__("sklearn.metrics", fromlist=["recall_score"])
            .recall_score(y_test, y_pred, average="weighted",
                          zero_division=0)), 4),
        "F1 Score" : round(float(
            f1_score(y_test, y_pred, average="weighted",
                     zero_division=0)), 4),
    })

metrics_df      = pd.DataFrame(ml_results).set_index("Model")
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score"]
x     = np.arange(len(MODEL_NAMES))
width = 0.2
fig, ax = plt.subplots(figsize=(11, 5))
bar_colors = ["#378ADD", "#1D9E75", "#D85A30", "#BA7517"]
for i, metric in enumerate(metrics_to_plot):
    vals = [metrics_df.loc[n, metric] for n in MODEL_NAMES]
    bars = ax.bar(x + i * width, vals, width,
                  label=metric, color=bar_colors[i], edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(MODEL_NAMES, rotation=10)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score")
ax.set_title("Overall Metrics — ML Baselines",
             fontsize=13, fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("outputs/ml/ml_overall_metrics.png",
            dpi=130, bbox_inches="tight")
plt.close()
print("✓ ml_overall_metrics.png saved")

# ── Plot 4: One-vs-Rest ROC curves ────────────────────────────
y_bin = label_binarize(y_test, classes=[0, 1, 2])
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, model, name in zip(axes, MODELS, MODEL_NAMES):
    _, y_prob = get_preds_probs(model, X_te_tfidf)
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})",
                color=COLORS[i], linewidth=1.8)
    ax.plot([0,1],[0,1],"k--", linewidth=0.8)
    ax.set_title(f"{name}\nOne-vs-Rest ROC", fontsize=10, fontweight="bold")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=8)
plt.suptitle("ROC Curves — ML Baselines",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/ml/ml_roc_curves.png",
            dpi=130, bbox_inches="tight")
plt.close()
print("✓ ml_roc_curves.png saved")

# ── Plot 5: Precision-Recall curves ──────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, model, name in zip(axes, MODELS, MODEL_NAMES):
    _, y_prob = get_preds_probs(model, X_te_tfidf)
    for i, cls in enumerate(CLASSES):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_bin[:, i], y_prob[:, i])
        ax.plot(rec, prec, label=f"{cls} (AP={ap:.2f})",
                color=COLORS[i], linewidth=1.8)
    ax.set_title(f"{name}\nPrecision-Recall", fontsize=10, fontweight="bold")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
plt.suptitle("Precision-Recall Curves — ML Baselines",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/ml/ml_pr_curves.png",
            dpi=130, bbox_inches="tight")
plt.close()
print("✓ ml_pr_curves.png saved")

# ── Plot 6: Top-20 TF-IDF features per class (LR only) ───────
fig, axes = plt.subplots(1, 3, figsize=(16, 7))
for ax, i, cls in zip(axes, range(len(CLASSES)), CLASSES):
    coef      = lr.coef_[i]
    top_idx   = np.argsort(coef)[-20:]
    top_words = np.array(tfidf.get_feature_names_out())[top_idx]
    top_vals  = coef[top_idx]
    ax.barh(top_words, top_vals, color=COLORS[i], edgecolor="white")
    ax.set_title(f"Top 20 features → {cls}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("LR Coefficient")
plt.suptitle("Logistic Regression — Top TF-IDF Features per Class",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/ml/ml_top_features.png",
            dpi=130, bbox_inches="tight")
plt.close()
print("✓ ml_top_features.png saved")

# ── Cleanup ───────────────────────────────────────────────────
del nb, lr, svm, svm_base, X_tr_tfidf, X_te_tfidf, tfidf
gc.collect()
print("\n✓ Cell 7a complete — all ML plots saved to outputs/ml/")
