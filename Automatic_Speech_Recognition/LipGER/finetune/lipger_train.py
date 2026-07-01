import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

# =============================
# CONFIG
# =============================
TRAIN_PATH = r"E:\ASR\facestar_whisper\facestar_full_train_whisper_fixed_clean.pt"
TEST_PATH = r"E:\ASR\facestar_whisper\facestar_full_test_whisper_fixed_clean.pt"
OUTPUT_DIR = r"E:\ASR\LipGER_trained_model"
MODEL_PATH = r"E:\ASR\TinyLlama\open_llama_3b"  # TinyLlama 3B local path

BATCH_SIZE = 32  # ✅ Changed batch size
EPOCHS = 2
LR = 2e-5
MAX_LEN = 512
GRAD_CLIP = 1.0
USE_TEST = True  # Set False agar sirf train chahiye

# =============================
# DEVICE
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================
# DATASET
# =============================
class LipGERDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=512):
        self.data = torch.load(path, map_location="cpu")
        self.tokenizer = tokenizer
        self.max_len = max_len
        assert isinstance(self.data, list), "PT file must contain a list of dicts"
        print(f"[INFO] Loaded {len(self.data)} samples from {path}")

    def build_prompt(self, nbest):
        best = nbest[0]
        others = "\n".join([f"{i}. {t}" for i, t in enumerate(nbest[1:], 1)])
        return f"Best hypothesis:\n{best}\n\nOther hypotheses:\n{others}\n\nCorrect transcription:"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        nbest = item.get("nhyps_base", [])
        if not nbest:
            nbest = [item.get("Caption", "")]
        gt = item.get("Caption", "")

        prompt = self.build_prompt(nbest)
        full_text = prompt + " " + gt

        enc = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# =============================
# TOKENIZER
# =============================
print("[INFO] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

# =============================
# BASE MODEL + LoRA
# =============================
print("[INFO] Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
base_model.gradient_checkpointing_enable()
base_model.enable_input_require_grads()

print("[INFO] Applying LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
model.train()

# =============================
# LOAD DATA
# =============================
train_dataset = LipGERDataset(TRAIN_PATH, tokenizer, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

if USE_TEST:
    test_dataset = LipGERDataset(TEST_PATH, tokenizer, MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =============================
# OPTIMIZER
# =============================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# =============================
# TRAINING LOOP
# =============================
for epoch in range(EPOCHS):
    print(f"[INFO] Starting epoch {epoch+1}/{EPOCHS}")
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"[EPOCH {epoch+1}] Avg Loss: {avg_loss:.4f}")

    # Save per epoch
    epoch_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}")
    os.makedirs(epoch_dir, exist_ok=True)
    model.save_pretrained(epoch_dir)
    tokenizer.save_pretrained(epoch_dir)

# =============================
# OPTIONAL: EVALUATION
# =============================
if USE_TEST:
    model.eval()
    print("[INFO] Starting evaluation on test set...")
    total_test_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            total_test_loss += outputs.loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"[TEST] Avg Loss: {avg_test_loss:.4f}")

# =============================
# SAVE FINAL MODEL
# =============================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("[INFO] Training completed successfully. Model saved at:", OUTPUT_DIR)