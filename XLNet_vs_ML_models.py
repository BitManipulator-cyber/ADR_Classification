# ============================================================
# CELL 7e — XLNet vs Naïve Bayes, LR, SVM
# ============================================================

import os, gc, json, re, warnings
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve,
    average_precision_score, f1_score,
    accuracy_score, precision_score, recall_score,
)
from sklearn.preprocessing import label_binarize
from transformers import (
    XLNetTokenizer,
    XLNetForSequenceClassification,
    logging as hf_logging,
)
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"
os.makedirs("outputs/comparison_xlnet", exist_ok=True)

SEED   = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(SEED)
torch.manual_seed(SEED)

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()

print(f"Device : {DEVICE}")
print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ── Step 1: Rebuild dataset from CSV ─────────────────────────
print("\nRebuilding dataset from CSV...")
CHUNK       = 100_000
SAMPLE_SIZE = 50_000
STOP_WORDS  = set(stopwords.words("english"))
chunks      = []

for chunk in pd.read_csv("/kaggle/input/datasets/amritanshukush/adverse-drug-reaction-adr-reporting/synthetic_drug_data.csv", chunksize=CHUNK,
                          dtype={
                              "PatientAge"      : "float32",
                              "DurationDays"    : "float32",
                              "OnsetDays"       : "float32",
                              "DrugName"        : "string",
                              "Dosage"          : "string",
                              "ConcomitantDrugs": "string",
                              "Seriousness"     : "string",
                          }):
    chunk["Seriousness"] = chunk["Seriousness"].str.strip().str.lower()
    chunk = chunk[chunk["Seriousness"].isin(["mild", "moderate", "severe"])]
    chunks.append(chunk)

df_full = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()

df, _ = train_test_split(df_full, train_size=SAMPLE_SIZE,
                         random_state=SEED, stratify=df_full["Seriousness"])
df = df.reset_index(drop=True)
del df_full
gc.collect()

df["ConcomitantDrugs"] = df["ConcomitantDrugs"].fillna("none")
df["DrugName"]         = df["DrugName"].fillna("unknown")
df["Dosage"]           = df["Dosage"].fillna("0mg")
df["PatientAge"]       = df["PatientAge"].fillna(df["PatientAge"].median())
df["DurationDays"]     = df["DurationDays"].fillna(df["DurationDays"].median())
df["OnsetDays"]        = df["OnsetDays"].fillna(df["OnsetDays"].median())

label_encoder           = LabelEncoder()
label_encoder.classes_  = np.array(["mild", "moderate", "severe"])
df["Label"]             = label_encoder.transform(df["Seriousness"])

def extract_dose(d):
    m = re.search(r"(\d+\.?\d*)", str(d))
    return m.group(1) if m else "0"

def build_text(row):
    drugs = re.sub(r"\s+", " ",
            str(row["ConcomitantDrugs"]).replace(",", " ").lower()).strip()
    return (f"drug {str(row['DrugName']).lower()} "
            f"dose {extract_dose(row['Dosage'])} mg "
            f"duration {int(row['DurationDays'])} days "
            f"age {int(row['PatientAge'])} "
            f"onset {int(row['OnsetDays'])} days "
            f"concomitant {drugs}")

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return " ".join(t for t in text.split()
                    if t not in STOP_WORDS and len(t) > 1)

df["combined_text"] = df.apply(build_text, axis=1)
df["cleaned_text"]  = df["combined_text"].apply(clean_text)

X_raw   = df["combined_text"].values.tolist()
X_clean = df["cleaned_text"].values.tolist()
y       = df["Label"].values.astype(np.int8)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=SEED, stratify=y)
X_train_clean, X_test_clean, _, _ = train_test_split(
    X_clean, y, test_size=0.2, random_state=SEED, stratify=y)

NUM_LABELS    = 3
CLASSES       = label_encoder.classes_
cw            = compute_class_weight("balanced",
                    classes=np.array([0, 1, 2]), y=y_train)
CLASS_WEIGHTS = torch.tensor(cw, dtype=torch.float).to(DEVICE)
del df
gc.collect()
print(f"✓ Dataset rebuilt — train:{len(X_train_raw)} test:{len(X_test_raw)}")

# ── PyTorch Dataset ───────────────────────────────────────────
class DrugADRDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        enc  = self.tokenizer(self.texts[idx], truncation=True,
                              padding="max_length", max_length=self.max_len,
                              return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item

# ── Step 2: Train ML models ───────────────────────────────────
print("\nTraining ML models...")
tfidf      = TfidfVectorizer(ngram_range=(1, 2), max_features=8000,
                              sublinear_tf=True)
X_tr_tfidf = tfidf.fit_transform(X_train_clean)
X_te_tfidf = tfidf.transform(X_test_clean)

nb = MultinomialNB()
nb.fit(X_tr_tfidf, y_train)
print("  ✓ Naïve Bayes")

lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                        random_state=SEED, solver="saga", n_jobs=-1)
lr.fit(X_tr_tfidf, y_train)
print("  ✓ Logistic Regression")

svm_base = LinearSVC(class_weight="balanced", random_state=SEED, max_iter=2000)
svm      = CalibratedClassifierCV(svm_base, cv=3)
svm.fit(X_tr_tfidf, y_train)
print("  ✓ SVM")

def get_ml_preds_probs(model, X_te):
    y_pred = model.predict(X_te)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_te)
    else:
        df2    = model.decision_function(X_te)
        df2    = df2 - df2.max(axis=1, keepdims=True)
        e      = np.exp(df2)
        y_prob = e / e.sum(axis=1, keepdims=True)
    return y_pred, y_prob

nb_preds,  nb_probs  = get_ml_preds_probs(nb,  X_te_tfidf)
lr_preds,  lr_probs  = get_ml_preds_probs(lr,  X_te_tfidf)
svm_preds, svm_probs = get_ml_preds_probs(svm, X_te_tfidf)
print("✓ ML predictions done")

# ── Step 3: Load XLNet results ────────────────────────────────
print("\nLoading XLNet saved results...")
xl_labels  = np.load("outputs/xl_labels.npy")
xl_preds   = np.load("outputs/xl_preds.npy")
xl_metrics = json.load(open("outputs/xl_metrics.json"))
print(f"✓ XLNet F1={xl_metrics['F1 Score']} | "
      f"Accuracy={xl_metrics['Accuracy']}")

# ── Step 4: XLNet probabilities ───────────────────────────────
print("\nReloading XLNet model for probability scores...")
xl_tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
xl_model     = XLNetForSequenceClassification.from_pretrained(
    "xlnet-base-cased", num_labels=NUM_LABELS,
    attn_implementation="eager", ignore_mismatched_sizes=True,
).to(DEVICE)
xl_model.load_state_dict(
    torch.load("outputs/xlnet_drug_adr.pt", map_location=DEVICE))
xl_model.eval()

test_dl       = DataLoader(
    DrugADRDataset(X_test_raw, y_test, xl_tokenizer, 128),
    batch_size=16, num_workers=2, pin_memory=True)
xl_probs_list = []
with torch.no_grad():
    for batch in test_dl:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        extra = {}
        if "token_type_ids" in batch:
            extra["token_type_ids"] = batch["token_type_ids"].to(DEVICE)
        with autocast():
            logits = xl_model(input_ids=ids, attention_mask=mask, **extra).logits
        xl_probs_list.extend(
            torch.softmax(logits, dim=1).cpu().numpy())
xl_probs = np.array(xl_probs_list)
del xl_model, test_dl, xl_tokenizer, xl_probs_list
free_memory()
print("✓ XLNet probabilities computed")

# ── Consolidate ───────────────────────────────────────────────
MODEL_NAMES = ["XLNet", "Naïve Bayes", "Logistic Regression", "SVM"]
COLORS_4    = ["#E07B39", "#1D9E75", "#D85A30", "#7F77DD"]
all_preds   = [xl_preds,  nb_preds,  lr_preds,  svm_preds]
all_probs   = [xl_probs,  nb_probs,  lr_probs,  svm_probs]
all_labels  = [xl_labels, y_test,    y_test,    y_test   ]
y_bin_test  = label_binarize(y_test,    classes=[0, 1, 2])
y_bin_xl    = label_binarize(xl_labels, classes=[0, 1, 2])
y_bins      = [y_bin_xl, y_bin_test, y_bin_test, y_bin_test]
roc_colors  = ["#378ADD", "#1D9E75", "#D85A30"]

def get_metrics(name, labels, preds):
    return {
        "Model"    : name,
        "Accuracy" : round(accuracy_score(labels, preds), 4),
        "Precision": round(precision_score(labels, preds,
                           average="weighted", zero_division=0), 4),
        "Recall"   : round(recall_score(labels, preds,
                           average="weighted", zero_division=0), 4),
        "F1 Score" : round(f1_score(labels, preds,
                           average="weighted", zero_division=0), 4),
    }

metrics_list = [
    get_metrics("XLNet",              xl_labels, xl_preds ),
    get_metrics("Naïve Bayes",        y_test,    nb_preds ),
    get_metrics("Logistic Regression",y_test,    lr_preds ),
    get_metrics("SVM",                y_test,    svm_preds),
]
metrics_df  = pd.DataFrame(metrics_list).set_index("Model")
metric_cols = ["Accuracy", "Precision", "Recall", "F1 Score"]
print("\nMetrics Summary:")
print(metrics_df.to_string())
metrics_df.to_csv("outputs/comparison_xlnet/xlnet_vs_ml_metrics.csv")
print("✓ xlnet_vs_ml_metrics.csv saved")

# ── Plot 1: Overall metrics grouped bar ───────────────────────
x     = np.arange(len(MODEL_NAMES))
width = 0.2
fig, ax = plt.subplots(figsize=(13, 5))
bc = ["#E07B39", "#1D9E75", "#D85A30", "#7F77DD"]
for i, metric in enumerate(metric_cols):
    vals = [metrics_df.loc[n, metric] for n in MODEL_NAMES]
    bars = ax.bar(x + i * width, vals, width,
                  label=metric, color=bc[i], edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(MODEL_NAMES, rotation=10)
ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
ax.set_title("XLNet vs ML Models — Overall Metrics",
             fontsize=13, fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("outputs/comparison_xlnet/xl_vs_ml_overall_metrics.png",
            dpi=130, bbox_inches="tight")
plt.close()
print("✓ xl_vs_ml_overall_metrics.png")

# ── Plot 2: F1 ranking horizontal bar ────────────────────────
sorted_df = metrics_df.sort_values("F1 Score", ascending=True)
fig, ax   = plt.subplots(figsize=(9, 4))
cs = ["#E07B39" if n == "XLNet" else "#888780" for n in sorted_df.index]
bars = ax.barh(sorted_df.index, sorted_df["F1 Score"],
               color=cs, edgecolor="white", height=0.5)
for bar, val in zip(bars, sorted_df["F1 Score"]):
    ax.text(bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=10)
ax.set_xlim(0, 1.15)
ax.set_title("F1 Score Ranking — XLNet vs ML Models",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Weighted F1 Score")
plt.tight_layout()
plt.savefig("outputs/comparison_xlnet/xl_vs_ml_f1_ranking.png",
            dpi=130, bbox_inches="tight")
plt.close()
print("✓ xl_vs_ml_f1_ranking.png")

# ── Plot 3: Per-class F1 ─────────────────────────────────────
per_cls = {name: f1_score(labels, preds, average=None,
                           labels=[0, 1, 2], zero_division=0)
           for name, labels, preds in
           zip(MODEL_NAMES, all_labels, all_preds)}

x     = np.arange(len(CLASSES))
width = 0.2
fig, ax = plt.subplots(figsize=(11, 5))
for i, name in enumerate(MODEL_NAMES):
    bars = ax.bar(x + i * width, per_cls[name], width,
                  label=name, color=COLORS_4[i], edgecolor="white")
    for bar, val in zip(bars, per_cls[name]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(CLASSES)
ax.set_ylim(0, 1.15); ax.set_ylabel("F1 Score")
ax.set_title("Per-class F1 — XLNet vs ML Models",
             fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/comparison_xlnet/xl_vs_ml_per_class_f1.png",
            dpi=130, bbox_inches="tight")
plt.close()
print("✓ xl_vs_ml_per_class_f1.png")

# ── Plot 4: Confusion matrices ────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
cmaps = ["Oranges", "Greens", "Reds", "Purples"]
for ax, name, labels, preds, cmap in zip(
        axes, MODEL_NAMES, all_labels, all_preds, cmaps):
    cm = confusion_matrix(labels, preds)
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(
        ax=ax, colorbar=False, cmap=cmap)
    ax.set_title(name, fontsize=10, fontweight="bold")
plt.suptitle("Confusion Matrices — XLNet vs ML Models",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/comparison_xlnet/xl_vs_ml_confusion_matrices.png",
            dpi=130, bbox_inches="tight")
plt.close()
print("✓ xl_vs_ml_confusion_matrices.png")

# ── Plot 5: ROC curves ────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, name, yb, probs in zip(axes, MODEL_NAMES, y_bins, all_probs):
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(yb[:, i], probs[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{cls} ({roc_auc:.2f})",
                color=roc_colors[i], linewidth=1.8)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_title(f"{name}\nROC (OvR)", fontsize=9, fontweight="bold")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=7)
plt.suptitle("ROC Curves — XLNet vs ML Models",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/comparison_xlnet/xl_vs_ml_roc_curves.png",
            dpi=130, bbox_inches="tight")
plt.close()
print("✓ xl_vs_ml_roc_curves.png")

# ── Plot 6: Precision-Recall curves ──────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, name, yb, probs in zip(axes, MODEL_NAMES, y_bins, all_probs):
    for i, cls in enumerate(CLASSES):
        prec, rec, _ = precision_recall_curve(yb[:, i], probs[:, i])
        ap = average_precision_score(yb[:, i], probs[:, i])
        ax.plot(rec, prec, label=f"{cls} ({ap:.2f})",
                color=roc_colors[i], linewidth=1.8)
    ax.set_title(f"{name}\nPR Curve", fontsize=9, fontweight="bold")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=7)
plt.suptitle("Precision-Recall Curves — XLNet vs ML Models",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/comparison_xlnet/xl_vs_ml_pr_curves.png",
            dpi=130, bbox_inches="tight")
plt.close()
print("✓ xl_vs_ml_pr_curves.png")

# ── Plot 7: Metrics heatmap ───────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
hm      = metrics_df[metric_cols].values.astype(float)
im      = ax.imshow(hm, cmap="YlOrBr", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(metric_cols))); ax.set_xticklabels(metric_cols)
ax.set_yticks(range(len(MODEL_NAMES))); ax.set_yticklabels(MODEL_NAMES)
for i in range(len(MODEL_NAMES)):
    for j in range(len(metric_cols)):
        ax.text(j, i, f"{hm[i,j]:.3f}",
                ha="center", va="center", fontsize=11)
plt.colorbar(im, ax=ax, fraction=0.03)
ax.set_title("Metrics Heatmap — XLNet vs ML Models",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/comparison_xlnet/xl_vs_ml_metrics_heatmap.png",
            dpi=130, bbox_inches="tight")
plt.close()
print("✓ xl_vs_ml_metrics_heatmap.png")

# ── Cleanup ───────────────────────────────────────────────────
del nb, lr, svm, svm_base, X_tr_tfidf, X_te_tfidf, tfidf
del xl_probs, nb_probs, lr_probs, svm_probs
del xl_preds, nb_preds, lr_preds, svm_preds
gc.collect()
print(f"\nGPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print("\n✓ Cell 7e complete — outputs/comparison_xlnet/")
