import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, BertForSequenceClassification,
    BartForConditionalGeneration, PreTrainedTokenizerFast
)
from sklearn.metrics import f1_score, precision_score, recall_score
import evaluate
from tqdm import tqdm

# ==============================
#  0. GPU 설정
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

# ==============================
#  1. 데이터 로드
# ==============================
def load_docs_for_abstractive(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = data["documents"]
    records = []
    for doc in docs:
        sents = [s["sentence"] for para in doc["text"] for s in para]
        abstr = doc["abstractive"][0]
        records.append({"text": " ".join(sents), "summary": abstr})
    return records

def load_docs_for_extractive(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = data["documents"]
    samples = []
    for doc in docs:
        sentences = [s["sentence"] for para in doc["text"] for s in para]
        labels = [1 if i in doc["extractive"] else 0 for i in range(len(sentences))]
        for sent, lbl in zip(sentences, labels):
            samples.append((sent, lbl))
    return samples

train_abs_records = load_docs_for_abstractive("data_aihub/train_original.json")
valid_abs_records = load_docs_for_abstractive("data_aihub/valid_original.json")
train_ext_samples = load_docs_for_extractive("data_aihub/train_original.json")
valid_ext_samples = load_docs_for_extractive("data_aihub/valid_original.json")

print(f" Loaded: {len(train_abs_records)} abstractive train / {len(valid_abs_records)} valid")
print(f" Loaded: {len(train_ext_samples)} extractive train / {len(valid_ext_samples)} valid")

# ==============================
#  2. Dataset 정의
# ==============================
# ---- 추출 요약 ----
MODEL_EXT = "./kobert"
tok_ext = AutoTokenizer.from_pretrained(MODEL_EXT)

class ExtDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len=128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sent, label = self.samples[idx]
        enc = self.tokenizer(
            sent, truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

train_ext_ds = ExtDataset(train_ext_samples, tok_ext)
valid_ext_ds = ExtDataset(valid_ext_samples, tok_ext)

train_ext_loader = DataLoader(train_ext_ds, batch_size=128, shuffle=True)
valid_ext_loader = DataLoader(valid_ext_ds, batch_size=128, shuffle=False)

# ---- 추상 요약 ----
MODEL_ABS = "./kobart"
tok_abs = PreTrainedTokenizerFast.from_pretrained(MODEL_ABS)

class AbsDataset(Dataset):
    def __init__(self, records, tokenizer, max_input_len=512, max_output_len=128):
        self.inputs = [r["text"] for r in records]
        self.targets = [r["summary"] for r in records]
        self.tok = tokenizer
        self.max_in = max_input_len
        self.max_out = max_output_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        src = self.tok(
            self.inputs[idx],
            max_length=self.max_in,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        tgt = self.tok(
            self.targets[idx],
            max_length=self.max_out,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": src["input_ids"].squeeze(0),
            "attention_mask": src["attention_mask"].squeeze(0),
            "labels": tgt["input_ids"].squeeze(0),
        }

train_abs_ds = AbsDataset(train_abs_records, tok_abs)
valid_abs_ds = AbsDataset(valid_abs_records, tok_abs)

train_abs_loader = DataLoader(train_abs_ds, batch_size=2, shuffle=True)
valid_abs_loader = DataLoader(valid_abs_ds, batch_size=2, shuffle=False)

# ==============================
#  3. 모델 & 옵티마이저
# ==============================
# 추출 요약
from transformers import BertModel
import torch.nn as nn

class BertForExtSummarization(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS 토큰
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}

model_ext = BertForExtSummarization(MODEL_EXT, num_labels=2).to(device)
optimizer_ext = torch.optim.AdamW(model_ext.parameters(), lr=3e-5)
criterion_ext = nn.CrossEntropyLoss()
best_f1 = 0.0
epochs_no_improve_ext = 0
patience_ext = 3

# 추상 요약
model_abs = BartForConditionalGeneration.from_pretrained(MODEL_ABS).to(device)
optimizer_abs = torch.optim.AdamW(model_abs.parameters(), lr=5e-5)
rouge_metric = evaluate.load("rouge")
best_rougeL = 0.0
epochs_no_improve_abs = 0
patience_abs = 3

EPOCHS = 10

# ==============================
#  4. 통합 학습 루프
# ==============================
for epoch in range(EPOCHS):
    print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")

    # ---- 추출 요약 학습 ----
    model_ext.train()
    total_loss_ext = 0
    for batch in tqdm(train_ext_loader, desc="[Train Extractive]"):
        optimizer_ext.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model_ext(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion_ext(outputs['logits'], labels)
        loss.backward()
        optimizer_ext.step()
        total_loss_ext += loss.item()
    avg_loss_ext = total_loss_ext / len(train_ext_loader)
    print(f"[Extractive] Train Loss: {avg_loss_ext:.4f}")

    # ---- 추출 요약 검증 ----
    model_ext.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in valid_ext_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model_ext(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs['logits'], dim=-1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    f1 = f1_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    print(f"[Extractive] F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        epochs_no_improve_ext = 0
        torch.save(model_ext.state_dict(), "best_kobert_ext.pt")
        print(f" Best Extractive model saved (F1={best_f1:.4f})")
    else:
        epochs_no_improve_ext += 1
        if epochs_no_improve_ext >= patience_ext:
            print(" Early stopping triggered for Extractive model.")

    # ---- 추상 요약 학습 ----
    model_abs.train()
    total_loss_abs = 0
    for batch in tqdm(train_abs_loader, desc="[Train Abstractive]"):
        optimizer_abs.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model_abs(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer_abs.step()
        total_loss_abs += loss.item()
    avg_loss_abs = total_loss_abs / len(train_abs_loader)
    print(f"[Abstractive] Train Loss: {avg_loss_abs:.4f}")

    # ---- 추상 요약 검증 ----
    model_abs.eval()
    pred_texts, ref_texts = [], []
    with torch.no_grad():
        for batch in valid_abs_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            generated_ids = model_abs.generate(
                input_ids=input_ids,
                max_length=128,
                num_beams=4,
                no_repeat_ngram_size=2
            )
            preds = tok_abs.batch_decode(generated_ids, skip_special_tokens=True)
            refs = tok_abs.batch_decode(labels, skip_special_tokens=True)
            pred_texts.extend(preds)
            ref_texts.extend(refs)

    results = rouge_metric.compute(predictions=pred_texts, references=ref_texts)
    rouge1 = results["rouge1"]
    rougeL = results["rougeL"]
    print(f"[Abstractive] ROUGE-1: {rouge1:.4f}, ROUGE-L: {rougeL:.4f}")

    if rougeL > best_rougeL:
        best_rougeL = rougeL
        epochs_no_improve_abs = 0
        torch.save(model_abs.state_dict(), "best_kobart_abs.pt")
        print(f" Best Abstractive model saved (ROUGE-L={best_rougeL:.4f})")
    else:
        epochs_no_improve_abs += 1
        if epochs_no_improve_abs >= patience_abs:
            print(" Early stopping triggered for Abstractive model.")

print("\n Training finished.")
print(f"Best Extractive F1: {best_f1:.4f}")
print(f"Best Abstractive ROUGE-L: {best_rougeL:.4f}")
