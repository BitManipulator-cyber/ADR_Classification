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


