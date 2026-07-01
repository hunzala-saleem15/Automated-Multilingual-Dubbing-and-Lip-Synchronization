# 1.py - LipGER ready-to-run with default args
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import numpy as np

# -----------------------------
# Path setup
# -----------------------------
wd = Path(__file__).parent.resolve()
sys.path.insert(0, str(wd))

# -----------------------------
# LipGER imports
# -----------------------------
try:
    from lipger.config import Config
    from lipger.utils import Compose, Normalize, RandomCrop, check_valid_checkpoint_dir, LazyLoadHF, chunked_cross_entropy
except ImportError as e:
    print("ImportError:", e)
    print("Check if modules exist and sys.path is correct.")
    exit(1)

# -----------------------------
# Preprocess
# -----------------------------
def get_preprocess():
    return Compose([
        Normalize(0.0, 255.0),
        RandomCrop((88, 88)),
        Normalize(0.421, 0.165)
    ])

# -----------------------------
# Batch loader
# -----------------------------
max_seq_len = 512  # adjust as needed
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_mouthroi(path):
    import h5py
    with h5py.File(path, "r") as hf:
        return hf["video_frames"][:]

def get_batch(data, model, preprocess):
    idx = torch.randint(len(data), (1,)).item()
    item = data[idx]

    x = item["input_ids"][:max_seq_len].long().unsqueeze(0)
    y = item["labels"][:max_seq_len].long().unsqueeze(0)

    mouth = torch.from_numpy(load_mouthroi(item["mouthroi"])).float()
    mouth = preprocess(mouth)
    mouth = mouth.unsqueeze(0).unsqueeze(1)

    vocab = model.config.vocab_size
    x = torch.clamp(x, 0, vocab-1)

    return x.to(device), y.to(device), mouth.to(device)

# -----------------------------
# Simple GPT model (placeholder)
# -----------------------------
class DummyModel(nn.Module):
    def __init__(self, vocab_size=1000, emb_dim=32):
        super().__init__()
        self.config = type('', (), {})()
        self.config.vocab_size = vocab_size
        self.lm_head = nn.Linear(emb_dim, vocab_size)
        self.embedding = nn.Embedding(vocab_size, emb_dim)
    def forward(self, input_ids=None):
        x = self.embedding(input_ids)
        x = self.lm_head(x)
        return x

# -----------------------------
# Main
# -----------------------------
def main(args=None):
    # -----------------------------
    # Default args if none
    # -----------------------------
    if args is None:
        class Args:
            train_path = "E:/ASR/facestar_whisper/formatted_json/data/mciro_train_whisper_tiny.pt"
            checkpoint_dir = "E:/ASR/models/open_llama_3b"
            device = device
            epochs = 2
            lr = 0.0002
        args = Args()

    print(f"[INFO] Using train_path: {args.train_path}")
    print(f"[INFO] Using checkpoint_dir: {args.checkpoint_dir}")
    print(f"[INFO] Using device: {args.device}")

    # -----------------------------
    # Load data
    # -----------------------------
    if not Path(args.train_path).exists():
        print("[ERROR] train_path does not exist!")
        return

    train_data = torch.load(args.train_path, weights_only=False)
    preprocess = get_preprocess()

    # -----------------------------
    # Load model (dummy for test)
    # -----------------------------
    model = DummyModel()
    model.to(args.device)
    model.eval()

    # -----------------------------
    # Load one batch and forward pass
    # -----------------------------
    x, y, mouth = get_batch(train_data, model, preprocess)

    print("\n=== Batch Info ===")
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("mouth shape:", mouth.shape)
    print("x unique:", torch.unique(x))
    print("y unique:", torch.unique(y))

    with torch.no_grad():
        logits = model(input_ids=x)
    print("logits shape:", logits.shape)

    # -----------------------------
    # Sample loss
    # -----------------------------
    try:
        criterion = nn.CrossEntropyLoss()
        y_ce = y.view(-1)
        logits_ce = logits.view(-1, logits.shape[-1])
        loss = criterion(logits_ce, y_ce)
        print("Sample loss:", loss.item())
    except Exception as e:
        print("Error computing loss:", e)

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--device", type=str, default=device)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    # Defaults if missing
    if not args.train_path:
        args.train_path = "E:/ASR/facestar_whisper/formatted_json/data/mciro_train_whisper_tiny.pt"
    if not args.checkpoint_dir:
        args.checkpoint_dir = "E:/ASR/models/open_llama_3b"

    main(args)
