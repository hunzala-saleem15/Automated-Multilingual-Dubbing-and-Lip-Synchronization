"""
LipGER Inference + Evaluation Script
FINAL – CRASH-PROOF
PyTorch 2.6+ compatible
TinyLLaMA compatible
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from safetensors.torch import load_file
from evaluate import load
from transformers import PreTrainedTokenizerFast

# =========================================================
# PyTorch 2.6 SAFE GLOBAL FIX
# =========================================================
import torch.serialization
torch.serialization.add_safe_globals([np.ndarray])

# =========================================================
# CONFIG
# =========================================================
class Config:
    def __init__(self, **kw):
        self.n_embd = kw["n_embd"]
        self.n_layer = kw["n_layer"]
        self.n_head = kw["n_head"]
        self.block_size = kw["block_size"]
        self.padded_vocab_size = kw["padded_vocab_size"]
        self.norm_eps = kw.get("norm_eps", 1e-5)
        self.bias = kw.get("bias", True)
        self.head_size = self.n_embd // self.n_head

# =========================================================
# RoPE
# =========================================================
def build_rope(seq_len, dim, device, dtype):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device) / dim))
    pos = torch.arange(seq_len, device=device)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)

def apply_rope(x, cos, sin):
    B, T, H, D = x.shape
    cos = cos[:T, :D].unsqueeze(0).unsqueeze(2)
    sin = sin[:T, :D].unsqueeze(0).unsqueeze(2)

    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    cos = cos[..., ::2]
    sin = sin[..., ::2]

    return torch.cat([x1 * cos - x2 * sin,
                      x1 * sin + x2 * cos], dim=-1)

# =========================================================
# ATTENTION
# =========================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.cfg = cfg

    def forward(self, x, rope):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.cfg.n_head, self.cfg.head_size)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        cos, sin = rope
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)

# =========================================================
# BLOCK
# =========================================================
class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, eps=cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, eps=cfg.norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
        )

    def forward(self, x, rope):
        x = x + self.attn(self.ln1(x), rope)
        x = x + self.mlp(self.ln2(x))
        return x

# =========================================================
# GPT
# =========================================================
class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.wte = nn.Embedding(cfg.padded_vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln = nn.LayerNorm(cfg.n_embd, eps=cfg.norm_eps)
        self.head = nn.Linear(cfg.n_embd, cfg.padded_vocab_size, bias=False)
        self.cfg = cfg
        self.rope = None

    def forward(self, idx):
        B, T = idx.shape
        x = self.wte(idx)
        if self.rope is None or self.rope[0].size(0) < T:
            self.rope = build_rope(self.cfg.block_size,
                                   self.cfg.head_size,
                                   idx.device,
                                   x.dtype)
        for blk in self.blocks:
            x = blk(x, self.rope)
        return self.head(self.ln(x))

# =========================================================
# GENERATION
# =========================================================
@torch.no_grad()
def generate(model, idx, max_new=150, eos_id=2):
    out = idx
    for _ in range(max_new):
        logits = model(out)[:, -1]
        next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        out = torch.cat([out, next_tok], dim=1)
        if next_tok.item() == eos_id:
            break
    return out[:, idx.size(1):]

# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    torch.set_float32_matmul_precision("high")

    # 🔥 CONFIG
    lit_config_path = Path(r"E:\ASR\TinyLlama\open_llama_3b\lit_config.json")
    with open(lit_config_path) as f:
        cfg = Config(**json.load(f))

    model = GPT(cfg).to(device).to(dtype).eval()
    model.load_state_dict(load_file(args.model_path), strict=False)

    # 🔥 TOKENIZER SAFE
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = tokenizer.convert_tokens_to_ids("<PAD>")
    ignore_id = -100  # use separate ignore_index for CrossEntropyLoss

    # 🔥 SAFE torch.load
    with torch.serialization.safe_globals([np.ndarray]):
        data = torch.load(args.test_data, weights_only=False)

    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_id)
    wer_metric = load("wer")

    total_loss = 0.0
    preds, refs = [], []

    for s in tqdm(data, desc="Inference"):
        inp = torch.tensor(s["input_ids_no_response"]).detach().clone().unsqueeze(0).to(device)

        gt_text = s.get("ground_truth", s.get("response_text", ""))
        gt_ids = tokenizer(gt_text, return_tensors="pt").input_ids.to(device)

        logits = model(inp)
        L = min(logits.size(1), gt_ids.size(1))

        loss = loss_fn(
            logits[:, -L:].reshape(-1, logits.size(-1)),
            gt_ids[:, -L:].reshape(-1)
        )
        total_loss += loss.item()

        out = generate(model, inp, eos_id=tokenizer.eos_token_id)
        preds.append(tokenizer.decode(out[0], skip_special_tokens=True))
        refs.append(gt_text)

    print("\n✅ INFERENCE COMPLETE")
    print("Average Loss:", total_loss / len(data))
    print("WER:", wer_metric.compute(predictions=preds, references=refs))
    print("Sample Output:", preds[:2])

if __name__ == "__main__":
    main()
