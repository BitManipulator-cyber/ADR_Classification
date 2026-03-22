# ============================================================
# CELL 7c — Visualizations for XLNet
# Self-contained: loads weights + results from outputs/
# ============================================================

import os, gc, json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve,
    average_precision_score, f1_score,
    classification_report
)
from sklearn.preprocessing import label_binarize
from transformers import (
    XLNetTokenizer,
    XLNetForSequenceClassification,
    logging as hf_logging,
)
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

hf_logging.set_verbosity_error()
os.makedirs("outputs/xlnet", exist_ok=True)

CLASSES = label_encoder.classes_
COLORS  = ["#378ADD", "#1D9E75", "#D85A30"]

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()

# ── Load saved results ────────────────────────────────────────
print("Loading XLNet saved results...")
xl_labels  = np.load("outputs/xl_labels.npy")
xl_preds   = np.load("outputs/xl_preds.npy")
xl_metrics = json.load(open("outputs/xl_metrics.json"))
xl_curves  = json.load(open("outputs/xl_curves.json"))
xl_losses  = xl_curves["losses"]
xl_f1s     = xl_curves["f1s"]
print(f"✓ F1={xl_metrics['F1 Score']} | Accuracy={xl_metrics['Accuracy']}")

# ── Plot 1: Confusion Matrix ──────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(xl_labels, xl_preds)
ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(
    ax=ax, colorbar=False, cmap="Oranges")
ax.set_title("XLNet — Confusion Matrix",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/xlnet/xl_confusion_matrix.png",
            dpi=130, bbox_inches="tight")
plt.close()
print("✓ xl_confusion_matrix.png saved")

# ── Plot 2: Training Loss + Validation F1 ────────────────────
epochs_x = list(range(1, len(xl_losses) + 1))
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(epochs_x, xl_losses, "o-", color="#D85A30",
             linewidth=2, markersize=7)
axes[0].set_title("Training Loss per Epoch")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].set_xticks(epochs_x)
for x, y in zip(epochs_x, xl_losses):
    axes[0].annotate(f"{y:.4f}", (x, y),
                     textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=9)

axes[1].plot(epochs_x, xl_f1s, "o-", color="#BA7517",
             linewidth=2, markersize=7)
axes[1].set_title("Validation F1 per Epoch")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Weighted F1")
axes[1].set_xticks(epochs_x)
axes[1].set_ylim(0, 1)
for x, y in zip(epochs_x, xl_f1s):
    axes[1].annotate(f"{y:.4f}", (x, y),
                     textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=9)

plt.suptitle("XLNet — Training Curves",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/xlnet/xl_training_curves.png",
            dpi=130, bbox_inches="tight")
plt.close()
print("✓ xl_training_curves.png saved")

# ── Plot 3: Per-class F1 bar chart ────────────────────────────
per_cls_f1 = f1_score(xl_labels, xl_preds, average=None,
                      labels=[0, 1, 2], zero_division=0)
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(CLASSES, per_cls_f1, color=COLORS,
              edgecolor="white", width=0.5)
for bar, val in zip(bars, per_cls_f1):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_ylabel("F1 Score")
ax.set_title("XLNet — Per-class F1 Score",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/xlnet/xl_per_class_f1.png",
            dpi=130, bbox_inches="tight")
plt.close()
print("✓ xl_per_class_f1.png saved")

# ── Plot 4: Overall metrics horizontal bar ────────────────────
metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
metric_vals  = [xl_metrics[m] for m in metric_names]
fig, ax = plt.subplots(figsize=(8, 4))
bar_colors = ["#378ADD", "#1D9E75", "#D85A30", "#BA7517"]
bars = ax.barh(metric_names[::-1], metric_vals[::-1],
               color=bar_colors[::-1], edgecolor="white", height=0.5)
for bar, val in zip(bars, metric_vals[::-1]):
    ax.text(bar.get_width() + 0.005,
            bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=11)
ax.set_xlim(0, 1.15)
ax.set_title("XLNet — Overall Metrics",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/xlnet/xl_overall_metrics.png",
            dpi=130, bbox_inches="tight")
plt.close()
print("✓ xl_overall_metrics.png saved")

# ── Reload model to get probability scores ────────────────────
print("\nReloading XLNet weights for probability plots...")
xl_tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
xl_model     = XLNetForSequenceClassification.from_pretrained(
    "xlnet-base-cased",
    num_labels              = NUM_LABELS,
    attn_implementation     = "eager",
    ignore_mismatched_sizes = True,
)
xl_model.load_state_dict(
    torch.load("outputs/xlnet_drug_adr.pt", map_location=DEVICE)
)
xl_model = xl_model.to(DEVICE)
xl_model.eval()
print("✓ Weights loaded")

test_dl = DataLoader(
    DrugADRDataset(X_test_raw, y_test, xl_tokenizer, 128),
    batch_size=16, num_workers=2, pin_memory=True
)

all_probs, all_true = [], []
with torch.no_grad():
    for batch in test_dl:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        extra = {}
        if "token_type_ids" in batch:
            extra["token_type_ids"] = batch["token_type_ids"].to(DEVICE)
        with autocast():
            logits = xl_model(
                input_ids=ids, attention_mask=mask, **extra
            ).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_true.extend(batch["labels"].numpy())

all_probs = np.array(all_probs)
all_true  = np.array(all_true)
y_bin     = label_binarize(all_true, classes=[0, 1, 2])
print("✓ Probabilities computed")

# ── Plot 5: One-vs-Rest ROC curves ────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for i, cls in enumerate(CLASSES):
    fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
    roc_auc     = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.3f})",
            color=COLORS[i], linewidth=2)
ax.plot([0,1],[0,1],"k--", linewidth=0.8)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("XLNet — One-vs-Rest ROC Curves",
             fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/xlnet/xl_roc_curves.png",
            dpi=130, bbox_inches="tight")
plt.close()
print("✓ xl_roc_curves.png saved")

# ── Plot 6: Precision-Recall curves ──────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for i, cls in enumerate(CLASSES):
    prec, rec, _ = precision_recall_curve(y_bin[:, i], all_probs[:, i])
    ap = average_precision_score(y_bin[:, i], all_probs[:, i])
    ax.plot(rec, prec, label=f"{cls} (AP={ap:.3f})",
            color=COLORS[i], linewidth=2)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_ylim(0, 1.05)
ax.set_title("XLNet — Precision-Recall Curves",
             fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/xlnet/xl_pr_curves.png",
            dpi=130, bbox_inches="tight")
plt.close()
print("✓ xl_pr_curves.png saved")

# ── Plot 7: Predicted probability distributions ───────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for i, (ax, cls) in enumerate(zip(axes, CLASSES)):
    ax.hist(all_probs[:, i], bins=40,
            color=COLORS[i], edgecolor="white", alpha=0.85)
    ax.set_title(f"Predicted P({cls})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Probability"); ax.set_ylabel("Count")
    ax.set_xlim(0, 1)
plt.suptitle("XLNet — Predicted Probability Distributions",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/xlnet/xl_prob_distributions.png",
            dpi=130, bbox_inches="tight")
plt.close()
print("✓ xl_prob_distributions.png saved")

# ── Plot 8: Classification report heatmap ────────────────────
report = classification_report(
    xl_labels, xl_preds,
    target_names=CLASSES,
    output_dict=True,
    zero_division=0
)
report_df = pd.DataFrame(report).transpose().iloc[:3][
    ["precision", "recall", "f1-score", "support"]
]
fig, ax = plt.subplots(figsize=(8, 3))
im = ax.imshow(
    report_df[["precision", "recall", "f1-score"]].values.astype(float),
    cmap="YlOrBr", aspect="auto", vmin=0, vmax=1
)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["Precision", "Recall", "F1-Score"])
ax.set_yticks(range(len(CLASSES)))
ax.set_yticklabels(CLASSES)
for i in range(len(CLASSES)):
    for j, col in enumerate(["precision", "recall", "f1-score"]):
        ax.text(j, i,
                f"{report_df[col].iloc[i]:.3f}",
                ha="center", va="center",
                fontsize=12, color="black")
plt.colorbar(im, ax=ax, fraction=0.03)
ax.set_title("XLNet — Classification Report Heatmap",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/xlnet/xl_classification_heatmap.png",
            dpi=130, bbox_inches="tight")
plt.close()
print("✓ xl_classification_heatmap.png saved")

# ── Plot 9: RoBERTa vs XLNet side-by-side comparison ─────────
print("\nGenerating RoBERTa vs XLNet comparison plot...")
rob_metrics_loaded = json.load(open("outputs/rob_metrics.json"))
rob_labels_loaded  = np.load("outputs/rob_labels.npy")
rob_preds_loaded   = np.load("outputs/rob_preds.npy")

model_names  = ["RoBERTa", "XLNet"]
metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
rob_vals     = [rob_metrics_loaded[m] for m in metric_names]
xl_vals      = [xl_metrics[m]         for m in metric_names]

x     = np.arange(len(metric_names))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width/2, rob_vals, width,
               label="RoBERTa", color="#378ADD", edgecolor="white")
bars2 = ax.bar(x + width/2, xl_vals,  width,
               label="XLNet",   color="#D85A30", edgecolor="white")
for bar, val in zip(bars1, rob_vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9)
for bar, val in zip(bars2, xl_vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(metric_names)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score")
ax.set_title("RoBERTa vs XLNet — All Metrics Comparison",
             fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/xlnet/roberta_vs_xlnet_comparison.png",
            dpi=130, bbox_inches="tight")
plt.close()
print("✓ roberta_vs_xlnet_comparison.png saved")

# ── Plot 10: Per-class F1 comparison ─────────────────────────
rob_per_cls = f1_score(rob_labels_loaded, rob_preds_loaded,
                       average=None, labels=[0,1,2], zero_division=0)
xl_per_cls  = f1_score(xl_labels, xl_preds,
                       average=None, labels=[0,1,2], zero_division=0)

x     = np.arange(len(CLASSES))
width = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width/2, rob_per_cls, width,
               label="RoBERTa", color="#378ADD", edgecolor="white")
bars2 = ax.bar(x + width/2, xl_per_cls,  width,
               label="XLNet",   color="#D85A30", edgecolor="white")
for bar, val in zip(bars1, rob_per_cls):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9)
for bar, val in zip(bars2, xl_per_cls):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(CLASSES)
ax.set_ylim(0, 1.15)
ax.set_ylabel("F1 Score")
ax.set_title("RoBERTa vs XLNet — Per-class F1 Comparison",
             fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/xlnet/roberta_vs_xlnet_per_class_f1.png",
            dpi=130, bbox_inches="tight")
plt.close()
print("✓ roberta_vs_xlnet_per_class_f1.png saved")

# ── Cleanup ───────────────────────────────────────────────────
del xl_model, test_dl, xl_tokenizer, all_probs, all_true
free_memory()
print(f"GPU after cleanup: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print("\n✓ Cell 7c complete — all plots saved to outputs/xlnet/")
print("\n  All outputs summary:")
print("  outputs/ml/          — 6 ML baseline plots")
print("  outputs/roberta/     — 8 RoBERTa plots")
print("  outputs/xlnet/       — 10 XLNet plots (incl. 2 comparison plots)")
