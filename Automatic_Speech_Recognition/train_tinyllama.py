import sys
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

# -----------------------------
# 🔹 TinyLlama module path
# -----------------------------
sys.path.append(r"E:\ASR\TinyLlama")  # yaha tumhara lit_gpt folder hai
from lit_gpt.model import GPT
from lit_gpt.config import Config

# -----------------------------
# 🔹 Tokenizer + Checkpoints path
# -----------------------------
CHECKPOINT_FOLDER = r"C:\TinyLlama-1.1B-3T"  # C me tumhari TinyLlama files
TOKENIZER_FILE = os.path.join(CHECKPOINT_FOLDER, "tokenizer.json")

# -----------------------------
# 🔹 Dataset Class
# -----------------------------
class JSONVideoAudioDataset(Dataset):
    def __init__(self, json_path, seq_len=128):
        self.seq_len = seq_len
        self.data = []

        with open(json_path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        for entry in entries:
            video_path = entry.get("video", "")
            audio_path = entry.get("audio", "")

            if video_path and os.path.exists(video_path) and audio_path and os.path.exists(audio_path):
                # Simple char-level tokenization (file names)
                text = os.path.basename(video_path) + " " + os.path.basename(audio_path)
                tokens = self.simple_tokenizer(text)
                for i in range(0, len(tokens) - seq_len, seq_len):
                    self.data.append(tokens[i:i + seq_len])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx][:-1], dtype=torch.long)
        y = torch.tensor(self.data[idx][1:], dtype=torch.long)
        return x, y

    @staticmethod
    def simple_tokenizer(text):
        return [ord(c) for c in text]

# -----------------------------
# 🔹 Load Dataset
# -----------------------------
JSON_PATH = r"E:\ASR\facestar_whisper\facestar_full_test_whisper.json"
dataset = JSONVideoAudioDataset(JSON_PATH, seq_len=128)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# -----------------------------
# 🔹 Device Check
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -----------------------------
# 🔹 Load Model
# -----------------------------
# Agar tumne checkpoints download kiye hain, config.json yaha point kare
config_path = os.path.join(CHECKPOINT_FOLDER, "config.json")
with open(config_path, "r", encoding="utf-8") as f:
    config_dict = json.load(f)

config = Config(**config_dict)
model = GPT(config).to(device)
model.train()

# -----------------------------
# 🔹 Loss & Optimizer
# -----------------------------
criterion = CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-4)

# -----------------------------
# 🔹 Training Loop
# -----------------------------
EPOCHS = 2  # adjust as needed
for epoch in range(EPOCHS):
    running_loss = 0.0
    for step, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if step % 10 == 0:
            print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")

print("Training completed ✅")
