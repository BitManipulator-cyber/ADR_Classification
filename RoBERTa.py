# ============================================================
# CELL 6a — RoBERTa Fine-tuning
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
