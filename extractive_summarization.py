import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertModel
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

# ==============================
#  0. GPU 설정
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

# ==============================
#  1. 데이터 로드
# ==============================
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

train_ext_samples = load_docs_for_extractive("data_aihub/train_original.json")
valid_ext_samples = load_docs_for_extractive("data_aihub/valid_original.json")

print(f" Loaded: {len(train_ext_samples)} extractive train / {len(valid_ext_samples)} valid")

# ==============================
#  2. Dataset 정의
# ==============================
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

train_ext_loader = DataLoader(train_ext_ds, batch_size=256, shuffle=True)
valid_ext_loader = DataLoader(valid_ext_ds, batch_size=256, shuffle=False)

# ==============================
#  3. 모델 정의
# ==============================
class BertForExtSummarization(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS
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
patience_ext = 5
EPOCHS = 20

# ==============================
#  4. 학습 루프
# ==============================
for epoch in range(EPOCHS):
    print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")

    # ---- Train ----
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

    # ---- Validation ----
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
        print(f" Best model saved (F1={best_f1:.4f})")
    else:
        epochs_no_improve_ext += 1
        if epochs_no_improve_ext >= patience_ext:
            print(" Early stopping triggered.")
            break

print("\n Training finished.")
print(f"Best Extractive F1: {best_f1:.4f}")
