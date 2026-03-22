# ============================================================
# CELL 6b — XLNet Fine-tuning (FIXED)
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

EPOCHS      = 3
BATCH       = 8
ACCUM_STEPS = 4
MAX_LEN     = 128
LR_TRANS    = 2e-5
PATIENCE    = 2

# ── Load RoBERTa results ──────────────────────────────────────
print("\nLoading RoBERTa results from outputs/...")
rob_metrics = json.load(open("outputs/rob_metrics.json"))
rob_curves  = json.load(open("outputs/rob_curves.json"))
rob_losses  = rob_curves["losses"]
rob_f1s     = rob_curves["f1s"]
rob_labels  = np.load("outputs/rob_labels.npy")
rob_preds   = np.load("outputs/rob_preds.npy")
print(f"✓ RoBERTa F1={rob_metrics['F1 Score']} | Accuracy={rob_metrics['Accuracy']}")

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

    print(f"\n{'='*45}\nXLNet — Test Results\n{'='*45}")
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


# ── Load & train XLNet ────────────────────────────────────────
print("\n" + "="*45)
print("Fine-tuning XLNet")
print("="*45)

xlnet_tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
xlnet_model     = XLNetForSequenceClassification.from_pretrained(
    "xlnet-base-cased",
    num_labels              = NUM_LABELS,
    attn_implementation     = "eager",
    ignore_mismatched_sizes = True,
).to(DEVICE)

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

xlnet_model, xl_losses, xl_f1s = train_model(
    xlnet_model, train_dl_xl, test_dl_xl, "XLNet"
)
xl_metrics, xl_labels, xl_preds = evaluate_model(
    xlnet_model, test_dl_xl, "XLNet"
)

# ── Save all results ──────────────────────────────────────────
torch.save(xlnet_model.state_dict(), "outputs/xlnet_drug_adr.pt")
np.save("outputs/xl_labels.npy", xl_labels)
np.save("outputs/xl_preds.npy",  xl_preds)
with open("outputs/xl_metrics.json", "w") as f: json.dump(xl_metrics, f)
with open("outputs/xl_curves.json",  "w") as f:
    json.dump({"losses": xl_losses, "f1s": xl_f1s}, f)

print(f"\n✓ XLNet F1={xl_metrics['F1 Score']} | Accuracy={xl_metrics['Accuracy']}")
print("✓ All results saved to outputs/")

del xlnet_model, train_dl_xl, test_dl_xl, xlnet_tokenizer
free_memory()
print(f"GPU after cleanup: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print("\n✓ Cell 6b done — run Cell 7 now")
