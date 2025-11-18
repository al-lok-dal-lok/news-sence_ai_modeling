import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from tqdm import tqdm
import evaluate

# ==============================
#  GPU 설정
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
#  데이터 로드
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

train_abs_records = load_docs_for_abstractive("data_aihub/train_original.json")
valid_abs_records = load_docs_for_abstractive("data_aihub/valid_original.json")

print(f"Loaded {len(train_abs_records)} train / {len(valid_abs_records)} valid")

# ==============================
#  Dataset 정의
# ==============================
MODEL_ABS = "./kobart"
tok_abs = PreTrainedTokenizerFast.from_pretrained(MODEL_ABS)

class AbsDataset(Dataset):
    def __init__(self, records, tokenizer, max_input_len=512, max_output_len=256):
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

train_abs_loader = DataLoader(train_abs_ds, batch_size=128, shuffle=True)
valid_abs_loader = DataLoader(valid_abs_ds, batch_size=128, shuffle=False)

# ==============================
#  모델 & 옵티마이저
# ==============================
model_abs = BartForConditionalGeneration.from_pretrained(MODEL_ABS).to(device)
optimizer_abs = torch.optim.AdamW(model_abs.parameters(), lr=5e-5)

rouge_metric = evaluate.load("rouge")
best_rougeL = 0.0
epochs_no_improve_abs = 0
patience_abs = 3
EPOCHS = 10

# ==============================
#  학습 루프
# ==============================
for epoch in range(EPOCHS):
    print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")

    # ---- Train ----
    model_abs.train()
    total_loss_abs = 0
    for batch in tqdm(train_abs_loader, desc="[Train Abstractive]"):
        optimizer_abs.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model_abs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer_abs.step()
        total_loss_abs += loss.item()

    print(f"[Abstractive] Train Loss: {total_loss_abs / len(train_abs_loader):.4f}")

    # ---- Validation ----
    model_abs.eval()
    pred_texts, ref_texts = [], []
    with torch.no_grad():
        for batch in valid_abs_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            generated_ids = model_abs.generate(
                input_ids=input_ids,
                max_length=256,
                num_beams=1,
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
        torch.save(model_abs.state_dict(), "best_kobart_abs.pt")
        print(f" Best model saved (ROUGE-L={best_rougeL:.4f})")
        epochs_no_improve_abs = 0
    else:
        epochs_no_improve_abs += 1
        if epochs_no_improve_abs >= patience_abs:
            print(" Early stopping triggered.")
            break

print("\nTraining finished.")
print(f"Best Abstractive ROUGE-L: {best_rougeL:.4f}")
