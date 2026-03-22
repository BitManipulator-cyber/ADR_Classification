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
