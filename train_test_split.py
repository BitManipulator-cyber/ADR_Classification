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
