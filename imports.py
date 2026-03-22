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
