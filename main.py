# ============================================================
# CELL 1 — Install Dependencies
# ============================================================
!pip install transformers torch scikit-learn pandas numpy matplotlib seaborn nltk -q

# ============================================================
# CELL 2 — Imports & Global Configuration
# ============================================================
import os, re, warnings, gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                  # no GUI backend — saves memory
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, ConfusionMatrixDisplay
)
from sklearn.calibration import CalibratedClassifierCV
from copy import deepcopy

import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification,
    XLNetTokenizer, XLNetForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW

import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

warnings.filterwarnings("ignore")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_LABELS = 3           # mild=0, moderate=1, severe=2
SAMPLE_SIZE = 50_000     # stratified sample from full dataset

os.makedirs("outputs", exist_ok=True)
print(f"Device : {DEVICE}")
print(f"Sample : {SAMPLE_SIZE:,} rows")

def free_memory():
    """Call between heavy steps to reclaim CPU + GPU RAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================
# CELL 3 — MODULE 1: Data Loading & Preprocessing
# ============================================================
# Memory optimizations:
#   - Read CSV in chunks, then sample — never loads 1M rows fully into RAM
#   - Downcast numeric columns immediately after load
#   - Drop raw/redundant columns after feature engineering
#   - Use category dtype for low-cardinality string columns
# ============================================================

# ── Upload if needed ─────────────────────────────────────────
# from google.colab import files
# files.upload()    # upload synthetic_drug_data.csv

# ── Chunked read + stratified sample ─────────────────────────
print("Reading CSV in chunks and sampling...")
CHUNK = 100_000
chunks = []

for chunk in pd.read_csv("/kaggle/input/datasets/amritanshukush/adverse-drug-reaction-adr-reporting/synthetic_drug_data.csv", chunksize=CHUNK,
                          dtype={
                              "ReportID"       : "string",
                              "PatientAge"     : "float32",
                              "DrugName"       : "string",
                              "Dosage"         : "string",
                              "DurationDays"   : "float32",
                              "ConcomitantDrugs": "string",
                              "ADR_Code"       : "string",
                              "Seriousness"    : "string",
                              "OnsetDays"      : "float32",
                          }):
    # Normalize label early so we can stratify-sample
    chunk["Seriousness"] = chunk["Seriousness"].str.strip().str.lower()
    chunk = chunk[chunk["Seriousness"].isin(["mild","moderate","severe"])]
    chunks.append(chunk)

df_full = pd.concat(chunks, ignore_index=True)
del chunks
free_memory()
print(f"Full valid rows: {len(df_full):,}")
print(f"Seriousness dist:\n{df_full['Seriousness'].value_counts()}")

# ── Stratified 50K sample ─────────────────────────────────────
df, _ = train_test_split(
    df_full,
    train_size=SAMPLE_SIZE,
    random_state=SEED,
    stratify=df_full["Seriousness"]
)
df = df.reset_index(drop=True)
del df_full
free_memory()
print(f"\nWorking sample: {len(df):,} rows")

# ── Fill missing values ───────────────────────────────────────
df["ConcomitantDrugs"] = df["ConcomitantDrugs"].fillna("none")
df["DrugName"]         = df["DrugName"].fillna("unknown")
df["Dosage"]           = df["Dosage"].fillna("0mg")
df["PatientAge"]       = df["PatientAge"].fillna(df["PatientAge"].median())
df["DurationDays"]     = df["DurationDays"].fillna(df["DurationDays"].median())
df["OnsetDays"]        = df["OnsetDays"].fillna(df["OnsetDays"].median())

# ── Encode target ─────────────────────────────────────────────
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(["mild", "moderate", "severe"])
df["Label"] = label_encoder.transform(df["Seriousness"])

# ── Build combined text field ─────────────────────────────────
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

df["combined_text"] = df.apply(build_text, axis=1)

# ── Clean text for TF-IDF ─────────────────────────────────────
STOP_WORDS = set(stopwords.words("english"))

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return " ".join(t for t in text.split()
                    if t not in STOP_WORDS and len(t) > 1)

df["cleaned_text"] = df["combined_text"].apply(clean_text)

# ── Drop columns no longer needed ────────────────────────────
df.drop(columns=["ReportID", "ADR_Code", "Dosage"], inplace=True)
free_memory()

# ── Class weights ─────────────────────────────────────────────
cw = compute_class_weight("balanced", classes=np.array([0,1,2]), y=df["Label"].values)
CLASS_WEIGHTS = torch.tensor(cw, dtype=torch.float).to(DEVICE)
print(f"\nClass weights → mild:{CLASS_WEIGHTS[0]:.3f} | "
      f"moderate:{CLASS_WEIGHTS[1]:.3f} | severe:{CLASS_WEIGHTS[2]:.3f}")

df.to_csv("outputs/cleaned_dataset.csv", index=False)
print(f"✓ Saved cleaned_dataset.csv  |  RAM used: {df.memory_usage(deep=True).sum()/1e6:.1f} MB")


# ============================================================
# CELL 4 — MODULE 2: Train/Test Split & PyTorch Dataset
# ============================================================
# Memory optimizations:
#   - Keep only the two text arrays + label array in RAM
#   - Delete the dataframe after extracting arrays
#   - Dataset tokenizes lazily per-batch (no full tensor in RAM)
# ============================================================

X_raw   = df["combined_text"].values.tolist()   # list is lighter than np array
X_clean = df["cleaned_text"].values.tolist()
y       = df["Label"].values.astype(np.int8)    # int8 saves vs int64

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=SEED, stratify=y
)
X_train_clean, X_test_clean, _, _ = train_test_split(
    X_clean, y, test_size=0.2, random_state=SEED, stratify=y
)

# Free the dataframe — no longer needed after this point
del df
free_memory()

print(f"Train: {len(X_train_raw):,}  |  Test: {len(X_test_raw):,}")
print(f"Train dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"Test  dist: {dict(zip(*np.unique(y_test,  return_counts=True)))}")

class DrugADRDataset(Dataset):
    """Tokenizes on-the-fly per __getitem__ — no giant tensor stored."""
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item

print("✓ Dataset class defined (lazy tokenization).")

# ============================================================
# CELL 5 — MODULE 3: Traditional ML Baselines (TF-IDF)
# ============================================================
# Memory optimizations:
#   - max_features=8000 (down from 10000) — smaller vocab matrix
#   - Delete TF-IDF matrix after each model fit/eval
#   - Use sparse matrix throughout (TfidfVectorizer default)
# ============================================================

tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=8000, sublinear_tf=True)
X_tr_tfidf = tfidf.fit_transform(X_train_clean)
X_te_tfidf = tfidf.transform(X_test_clean)
print(f"TF-IDF matrix: {X_tr_tfidf.shape}  "
      f"({X_tr_tfidf.data.nbytes / 1e6:.1f} MB sparse)")

def evaluate_sklearn(model, X_te, y_te, model_name):
    y_pred = model.predict(X_te)
    metrics = {
        "Model"    : model_name,
        "Accuracy" : round(accuracy_score(y_te, y_pred), 4),
        "Precision": round(precision_score(y_te, y_pred, average="weighted", zero_division=0), 4),
        "Recall"   : round(recall_score(y_te, y_pred, average="weighted", zero_division=0), 4),
        "F1 Score" : round(f1_score(y_te, y_pred, average="weighted", zero_division=0), 4),
    }
    print(f"\n{'='*40}\n{model_name}")
    print(classification_report(y_te, y_pred,
          target_names=label_encoder.classes_, zero_division=0))
    return metrics

all_results = []

# Naïve Bayes
nb = MultinomialNB()
nb.fit(X_tr_tfidf, y_train)
all_results.append(evaluate_sklearn(nb, X_te_tfidf, y_test, "Naïve Bayes"))
del nb
free_memory()

# Logistic Regression
lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                        random_state=SEED, solver="saga", n_jobs=-1)
lr.fit(X_tr_tfidf, y_train)
all_results.append(evaluate_sklearn(lr, X_te_tfidf, y_test, "Logistic Regression"))
del lr
free_memory()

# SVM
svm_base = LinearSVC(class_weight="balanced", random_state=SEED, max_iter=2000)
svm = CalibratedClassifierCV(svm_base, cv=3)
svm.fit(X_tr_tfidf, y_train)
all_results.append(evaluate_sklearn(svm, X_te_tfidf, y_test, "SVM (LinearSVC)"))
del svm, svm_base, X_tr_tfidf, X_te_tfidf, tfidf
free_memory()

print("\n✓ Traditional ML baselines complete.")

# ============================================================
# CELL 6a — RoBERTa Fine-tuning
# Run this first. When done, run Cell 6b immediately.
# DO NOT restart runtime between 6a and 6b.
# ============================================================

import os, gc, json
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from copy import deepcopy
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
    logging as hf_logging,
)
from torch.optim import AdamW

hf_logging.set_verbosity_error()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"
os.makedirs("outputs", exist_ok=True)

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

total = torch.cuda.get_device_properties(0).total_memory / 1e9
used  = torch.cuda.memory_allocated() / 1e9
print(f"GPU  : {torch.cuda.get_device_name(0)}")
print(f"VRAM : {total:.1f} GB total | {used:.1f} GB used | {total-used:.1f} GB free")

# ── Hyperparameters ───────────────────────────────────────────
EPOCHS      = 3
BATCH       = 8
ACCUM_STEPS = 4      # effective batch = 32
MAX_LEN     = 128
LR_TRANS    = 2e-5
PATIENCE    = 2

# ── Training function ─────────────────────────────────────────
def train_model(model, train_loader, val_loader, model_name):
    scaler      = GradScaler()
    optimizer   = AdamW(model.parameters(), lr=LR_TRANS, weight_decay=0.01)
    total_steps = (len(train_loader) // ACCUM_STEPS) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps  = max(1, int(0.06 * total_steps)),
        num_training_steps= total_steps,
    )
    loss_fn     = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
    train_losses, val_f1s = [], []
    best_f1, best_state   = 0.0, None
    no_improve            = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            ids  = batch["input_ids"].to(DEVICE, non_blocking=True)
            mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            lbls = batch["labels"].to(DEVICE, non_blocking=True)
            extra = {}
            if "token_type_ids" in batch:
                extra["token_type_ids"] = batch["token_type_ids"].to(DEVICE, non_blocking=True)

            with autocast():
                logits = model(input_ids=ids, attention_mask=mask, **extra).logits
                loss   = loss_fn(logits, lbls) / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * ACCUM_STEPS

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(round(avg_loss, 6))

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids  = batch["input_ids"].to(DEVICE, non_blocking=True)
                mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
                extra = {}
                if "token_type_ids" in batch:
                    extra["token_type_ids"] = batch["token_type_ids"].to(DEVICE, non_blocking=True)
                with autocast():
                    logits = model(input_ids=ids, attention_mask=mask, **extra).logits
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                trues.extend(batch["labels"].numpy())

        val_f1   = f1_score(trues, preds, average="weighted", zero_division=0)
        gpu_used = torch.cuda.memory_allocated() / 1e9
        val_f1s.append(round(val_f1, 6))
        print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} "
              f"| Val F1: {val_f1:.4f} | GPU: {gpu_used:.2f} GB")

        if val_f1 > best_f1 + 1e-4:
            best_f1    = val_f1
            best_state = deepcopy(model.state_dict())
            no_improve = 0
            print(f"  ✓ Checkpoint saved (F1={best_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stop — best F1: {best_f1:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        del best_state
        free_memory()

    return model, train_losses, val_f1s


# ── Evaluation function ───────────────────────────────────────
def evaluate_model(model, loader, model_name):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE, non_blocking=True)
            mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            extra = {}
            if "token_type_ids" in batch:
                extra["token_type_ids"] = batch["token_type_ids"].to(DEVICE, non_blocking=True)
            with autocast():
                logits = model(input_ids=ids, attention_mask=mask, **extra).logits
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(batch["labels"].numpy())

    print(f"\n{'='*45}\n{model_name} — Test Results\n{'='*45}")
    print(classification_report(
        all_labels, all_preds,
        target_names=label_encoder.classes_, zero_division=0
    ))
    return {
        "Model"    : model_name,
        "Accuracy" : round(accuracy_score(all_labels, all_preds), 4),
        "Precision": round(precision_score(all_labels, all_preds, average="weighted", zero_division=0), 4),
        "Recall"   : round(recall_score(all_labels, all_preds, average="weighted", zero_division=0), 4),
        "F1 Score" : round(f1_score(all_labels, all_preds, average="weighted", zero_division=0), 4),
    }, np.array(all_labels, dtype=np.int8), np.array(all_preds, dtype=np.int8)


# ── Load & train RoBERTa ──────────────────────────────────────
print("\n" + "="*45)
print("Fine-tuning RoBERTa")
print("="*45)

roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_model     = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels              = NUM_LABELS,
    attn_implementation     = "eager",
    ignore_mismatched_sizes = True,
).to(DEVICE)

train_dl_rob = DataLoader(
    DrugADRDataset(X_train_raw, y_train, roberta_tokenizer, MAX_LEN),
    batch_size=BATCH, shuffle=True,
    num_workers=2, pin_memory=True, persistent_workers=True
)
test_dl_rob = DataLoader(
    DrugADRDataset(X_test_raw, y_test, roberta_tokenizer, MAX_LEN),
    batch_size=BATCH * 2,
    num_workers=2, pin_memory=True, persistent_workers=True
)

roberta_model, rob_losses, rob_f1s = train_model(
    roberta_model, train_dl_rob, test_dl_rob, "RoBERTa"
)
rob_metrics, rob_labels, rob_preds = evaluate_model(
    roberta_model, test_dl_rob, "RoBERTa"
)

# ── Save all results to disk ──────────────────────────────────
torch.save(roberta_model.state_dict(), "outputs/roberta_drug_adr.pt")
np.save("outputs/rob_labels.npy", rob_labels)
np.save("outputs/rob_preds.npy",  rob_preds)
with open("outputs/rob_metrics.json", "w") as f: json.dump(rob_metrics, f)
with open("outputs/rob_curves.json",  "w") as f:
    json.dump({"losses": rob_losses, "f1s": rob_f1s}, f)

print(f"\n✓ RoBERTa F1={rob_metrics['F1 Score']} | Accuracy={rob_metrics['Accuracy']}")
print("✓ All results saved to outputs/")

# ── Wipe from GPU before XLNet ────────────────────────────────
del roberta_model, train_dl_rob, test_dl_rob, roberta_tokenizer
free_memory()
print(f"GPU after cleanup: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print("\n✓ Cell 6a done — run Cell 6b now")

# ============================================================
# CELL 6b — XLNet Fine-tuning ONLY
# Loads RoBERTa results from disk, trains XLNet fresh
# NO gradient_checkpointing — XLNet does not support it
# ============================================================

import os, gc, json
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from copy import deepcopy
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)
from transformers import (
    XLNetTokenizer,
    XLNetForSequenceClassification,
    get_linear_schedule_with_warmup,
    logging as hf_logging,
)
from torch.optim import AdamW

# ── Silence all HF warnings including LOAD REPORT ────────────
hf_logging.set_verbosity_error()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"
os.makedirs("outputs", exist_ok=True)

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# ── GPU status ────────────────────────────────────────────────
total = torch.cuda.get_device_properties(0).total_memory / 1e9
used  = torch.cuda.memory_allocated() / 1e9
print(f"GPU  : {torch.cuda.get_device_name(0)}")
print(f"VRAM : {total:.1f} GB total | {used:.1f} GB used | {total-used:.1f} GB free")

# ── Hyperparameters ───────────────────────────────────────────
EPOCHS      = 3
BATCH       = 8
ACCUM_STEPS = 4      # effective batch = 32
MAX_LEN     = 128
LR_TRANS    = 2e-5
PATIENCE    = 2

# ── Load RoBERTa results saved earlier ───────────────────────
print("\nLoading RoBERTa results...")
rob_metrics = json.load(open("outputs/rob_metrics.json"))
rob_curves  = json.load(open("outputs/rob_curves.json"))
rob_losses  = rob_curves["losses"]
rob_f1s     = rob_curves["f1s"]
rob_labels  = np.load("outputs/rob_labels.npy")
rob_preds   = np.load("outputs/rob_preds.npy")
print(f"✓ RoBERTa F1={rob_metrics['F1 Score']} | Accuracy={rob_metrics['Accuracy']}")

# ── Training function — NO gradient checkpointing ────────────
def train_model(model, train_loader, val_loader, model_name):

    # Explicitly make sure it's off — belt and suspenders
    model.config.use_cache = True          # restore default (was disabled by HF warning)
    if hasattr(model, "gradient_checkpointing_disable"):
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass                           # XLNet may raise — safely ignored

    scaler      = GradScaler()
    optimizer   = AdamW(model.parameters(), lr=LR_TRANS, weight_decay=0.01)
    total_steps = (len(train_loader) // ACCUM_STEPS) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps  = max(1, int(0.06 * total_steps)),
        num_training_steps= total_steps,
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)

    train_losses, val_f1s = [], []
    best_f1, best_state   = 0.0, None
    no_improve            = 0

    for epoch in range(EPOCHS):
        # ── Train ─────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            ids  = batch["input_ids"].to(DEVICE, non_blocking=True)
            mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            lbls = batch["labels"].to(DEVICE, non_blocking=True)
            extra = {}
            if "token_type_ids" in batch:
                extra["token_type_ids"] = batch["token_type_ids"].to(
                    DEVICE, non_blocking=True)

            with autocast():
                logits = model(
                    input_ids=ids, attention_mask=mask, **extra
                ).logits
                loss = loss_fn(logits, lbls) / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * ACCUM_STEPS

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(round(avg_loss, 6))

        # ── Validate ──────────────────────────────────────────
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids  = batch["input_ids"].to(DEVICE, non_blocking=True)
                mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
                extra = {}
                if "token_type_ids" in batch:
                    extra["token_type_ids"] = batch["token_type_ids"].to(
                        DEVICE, non_blocking=True)
                with autocast():
                    logits = model(
                        input_ids=ids, attention_mask=mask, **extra
                    ).logits
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                trues.extend(batch["labels"].numpy())

        val_f1   = f1_score(trues, preds, average="weighted", zero_division=0)
        gpu_used = torch.cuda.memory_allocated() / 1e9
        val_f1s.append(round(val_f1, 6))
        print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} "
              f"| Val F1: {val_f1:.4f} | GPU: {gpu_used:.2f} GB")

        if val_f1 > best_f1 + 1e-4:
            best_f1    = val_f1
            best_state = deepcopy(model.state_dict())
            no_improve = 0
            print(f"  ✓ Checkpoint saved (F1={best_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stop — best F1: {best_f1:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        del best_state
        free_memory()

    return model, train_losses, val_f1s


# ── Evaluation function ───────────────────────────────────────
def evaluate_model(model, loader, model_name):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE, non_blocking=True)
            mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            extra = {}
            if "token_type_ids" in batch:
                extra["token_type_ids"] = batch["token_type_ids"].to(
                    DEVICE, non_blocking=True)
            with autocast():
                logits = model(
                    input_ids=ids, attention_mask=mask, **extra
                ).logits
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(batch["labels"].numpy())

    print(f"\n{'='*45}\nXLNet — Test Results\n{'='*45}")
    print(classification_report(
        all_labels, all_preds,
        target_names=label_encoder.classes_, zero_division=0
    ))
    return {
        "Model"    : model_name,
        "Accuracy" : round(accuracy_score(all_labels, all_preds), 4),
        "Precision": round(precision_score(all_labels, all_preds,
                           average="weighted", zero_division=0), 4),
        "Recall"   : round(recall_score(all_labels, all_preds,
                           average="weighted", zero_division=0), 4),
        "F1 Score" : round(f1_score(all_labels, all_preds,
                           average="weighted", zero_division=0), 4),
    }, np.array(all_labels, dtype=np.int8), np.array(all_preds, dtype=np.int8)


# ── Load XLNet ────────────────────────────────────────────────
print("\n" + "="*45)
print("Fine-tuning XLNet")
print("="*45)

xlnet_tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
xlnet_model = XLNetForSequenceClassification.from_pretrained(
    "xlnet-base-cased",
    num_labels              = NUM_LABELS,
    attn_implementation     = "eager",
    ignore_mismatched_sizes = True,
).to(DEVICE)

# Confirm checkpointing is off — XLNet raises ValueError if you call enable()
print(f"Gradient checkpointing supported: "
      f"{xlnet_model.supports_gradient_checkpointing}")   # will print False
print(f"use_cache: {xlnet_model.config.use_cache}")       # should be True

train_dl_xl = DataLoader(
    DrugADRDataset(X_train_raw, y_train, xlnet_tokenizer, MAX_LEN),
    batch_size=BATCH, shuffle=True,
    num_workers=2, pin_memory=True, persistent_workers=True
)
test_dl_xl = DataLoader(
    DrugADRDataset(X_test_raw, y_test, xlnet_tokenizer, MAX_LEN),
    batch_size=BATCH * 2,
    num_workers=2, pin_memory=True, persistent_workers=True
)

# ── Train ─────────────────────────────────────────────────────
xlnet_model, xl_losses, xl_f1s = train_model(
    xlnet_model, train_dl_xl, test_dl_xl, "XLNet"
)

# ── Evaluate ──────────────────────────────────────────────────
xl_metrics, xl_labels, xl_preds = evaluate_model(
    xlnet_model, test_dl_xl, "XLNet"
)

# ── Save everything ───────────────────────────────────────────
torch.save(xlnet_model.state_dict(), "outputs/xlnet_drug_adr.pt")
np.save("outputs/xl_labels.npy", xl_labels)
np.save("outputs/xl_preds.npy",  xl_preds)
with open("outputs/xl_metrics.json", "w") as f: json.dump(xl_metrics, f)
with open("outputs/xl_curves.json",  "w") as f:
    json.dump({"losses": xl_losses, "f1s": xl_f1s}, f)

print(f"\n✓ XLNet F1={xl_metrics['F1 Score']} | Accuracy={xl_metrics['Accuracy']}")
print("  Saved: xlnet_drug_adr.pt | xl_labels.npy | xl_preds.npy")
print("  Saved: xl_metrics.json | xl_curves.json")

# ── Cleanup ───────────────────────────────────────────────────
del xlnet_model, train_dl_xl, test_dl_xl, xlnet_tokenizer
free_memory()
print(f"\nGPU after cleanup: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print("\n✓ Cell 6b complete — run Cell 7 now.")


# ============================================================
# CELL 7 — MODULES 5 & 6: Evaluation, Plots & Final Report
# ============================================================
# Memory optimizations:
#   - plt.close() after every figure — releases figure RAM
#   - Save directly to disk, never hold multiple open figures
#   - del model objects after plotting is done
# ============================================================

# ── Comparison Table ──────────────────────────────────────────
results_df = pd.DataFrame(all_results) \
               .sort_values("F1 Score", ascending=False) \
               .reset_index(drop=True)
print("\nFull Comparison Table:")
print(results_df.to_string(index=False))
results_df.to_csv("outputs/comparison_table.csv", index=False)
print("✓ comparison_table.csv saved.")

# ── Plot 1: F1 & Accuracy Bar Charts ─────────────────────────
colors = ["#8ecae6","#219ebc","#023047","#ffb703","#fb8500"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, metric in zip(axes, ["F1 Score", "Accuracy"]):
    bars = ax.bar(results_df["Model"], results_df[metric],
                  color=colors[:len(results_df)], edgecolor="white")
    ax.set_title(f"{metric} Comparison"); ax.set_ylim(0.0, 1.05)
    ax.set_xticklabels(results_df["Model"], rotation=20, ha="right")
    for bar, val in zip(bars, results_df[metric]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
plt.suptitle("Model Comparison — ADR Seriousness", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/metric_comparison.png", dpi=120, bbox_inches="tight")
plt.close(); free_memory()
print("✓ metric_comparison.png saved.")

# ── Plot 2: Transformer Training Curves ──────────────────────
epochs_x = range(1, len(rob_losses) + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(epochs_x, rob_losses, "o-", label="RoBERTa", color="#023047")
axes[0].plot(range(1, len(xl_losses)+1), xl_losses, "s-", label="XLNet", color="#fb8500")
axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
axes[1].plot(epochs_x, rob_f1s, "o-", label="RoBERTa", color="#023047")
axes[1].plot(range(1, len(xl_f1s)+1), xl_f1s, "s-", label="XLNet", color="#fb8500")
axes[1].set_title("Validation F1"); axes[1].set_xlabel("Epoch"); axes[1].legend()
plt.suptitle("Transformer Training Curves", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/training_curves.png", dpi=120, bbox_inches="tight")
plt.close(); free_memory()
print("✓ training_curves.png saved.")

# ── Plot 3: Confusion Matrices ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (preds, labels, name) in zip(axes, [
    (rob_preds, rob_labels, "RoBERTa"),
    (xl_preds,  xl_labels,  "XLNet")
]):
    ConfusionMatrixDisplay.from_predictions(
        labels, preds,
        display_labels=label_encoder.classes_,
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"{name} — Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/confusion_matrices.png", dpi=120, bbox_inches="tight")
plt.close(); free_memory()
print("✓ confusion_matrices.png saved.")

# ── Plot 4: Per-class F1 heatmap ──────────────────────────────
from sklearn.metrics import f1_score as f1_per

class_f1_data = {}
for name, (preds, labels) in [
    ("Naïve Bayes",       (None, None)),   # placeholder — recompute below
    ("Logistic Reg.",     (None, None)),
    ("SVM",               (None, None)),
    ("RoBERTa",           (rob_preds, rob_labels)),
    ("XLNet",             (xl_preds,  xl_labels)),
]:
    if preds is not None:
        scores = f1_score(labels, preds, average=None,
                          labels=[0,1,2], zero_division=0)
        class_f1_data[name] = scores

heatmap_df = pd.DataFrame(class_f1_data,
                           index=label_encoder.classes_).T
fig, ax = plt.subplots(figsize=(7, 3))
im = ax.imshow(heatmap_df.values, cmap="YlGn", aspect="auto",
               vmin=0, vmax=1)
ax.set_xticks(range(3)); ax.set_xticklabels(label_encoder.classes_)
ax.set_yticks(range(len(heatmap_df))); ax.set_yticklabels(heatmap_df.index)
for i in range(len(heatmap_df)):
    for j in range(3):
        ax.text(j, i, f"{heatmap_df.values[i,j]:.2f}",
                ha="center", va="center", fontsize=10, color="black")
plt.colorbar(im, ax=ax, fraction=0.03)
ax.set_title("Per-class F1 Score Heatmap (Transformer Models)")
plt.tight_layout()
plt.savefig("outputs/f1_heatmap.png", dpi=120, bbox_inches="tight")
plt.close(); free_memory()
print("✓ f1_heatmap.png saved.")

# ── Free model objects — no longer needed ─────────────────────
del roberta_model, xlnet_model
free_memory()

# ── Final Report ──────────────────────────────────────────────
best = results_df.iloc[0]
report = f"""
DRUG ADR SERIOUSNESS CLASSIFICATION — FINAL REPORT
===================================================
Date      : {pd.Timestamp.now().strftime("%Y-%m-%d")}
Dataset   : synthetic_drug_data.csv  (1,000,000 rows → 50,000 sampled)
Task      : Multiclass — mild (0) / moderate (1) / severe (2)

FEATURE ENGINEERING
-------------------
  Combined text: DrugName + Dosage + DurationDays +
                 PatientAge + OnsetDays + ConcomitantDrugs
  TF-IDF input : cleaned (lowercase, stopwords removed)
  Transformer  : raw combined text (preserves numeric context)

TRAIN / TEST SPLIT
------------------
  Train: 40,000  |  Test: 10,000  |  Stratified by Seriousness

MODEL RESULTS (sorted by Weighted F1)
--------------------------------------
{results_df.to_string(index=False)}

BEST MODEL : {best['Model']}
  Accuracy  : {best['Accuracy']}
  Precision : {best['Precision']}
  Recall    : {best['Recall']}
  F1 Score  : {best['F1 Score']}

OUTPUT FILES
------------
  outputs/cleaned_dataset.csv
  outputs/comparison_table.csv
  outputs/metric_comparison.png
  outputs/training_curves.png
  outputs/confusion_matrices.png
  outputs/f1_heatmap.png
  outputs/roberta_drug_adr.pt
  outputs/xlnet_drug_adr.pt
  outputs/final_report.txt
"""
with open("outputs/final_report.txt", "w") as f:
    f.write(report)
print(report)
print("✓ All 7 cells complete.")
