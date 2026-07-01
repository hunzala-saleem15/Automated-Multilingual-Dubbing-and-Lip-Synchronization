"""
LipGER Inference Script
Single-file GPT + Generate + KV-cache SAFE + RoPE SAFE + BF16 / AMP
"""

import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from safetensors.torch import load_file
from evaluate import load
import argparse
import h5py
import numpy as np

# ------------------------------
# CONFIG
# ------------------------------
class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.n_embd = kwargs.get("n_embd", 256)
        self.n_layer = kwargs.get("n_layer", 8)
        self.n_head = kwargs.get("n_head", 8)
        self.head_size = self.n_embd // self.n_head
        self.block_size = kwargs.get("block_size", 2048)
        self.padded_vocab_size = kwargs.get("padded_vocab_size", 32000)
        self.norm_class = nn.LayerNorm
        self.norm_eps = 1e-5
        self.rotary_percentage = 1.0
        self.bias = True
        self.parallel_residual = False

# ------------------------------
# RoPE helpers
# ------------------------------
def build_rope_cache(seq_len, n_elem, dtype, device, base=10000):
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device) / n_elem))
    seq_idx = torch.arange(seq_len, device=device)
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)
    return torch.cos(idx_theta).to(dtype), torch.sin(idx_theta).to(dtype)

def apply_rope(x, cos, sin):
    n = cos.size(-1)
    cos = cos[: x.size(1), :n].unsqueeze(0).unsqueeze(2)
    sin = sin[: x.size(1), :n].unsqueeze(0).unsqueeze(2)
    x1, x2 = x[..., :n], x[..., n:]
    x_rot = torch.cat((-x1[..., n // 2:], x1[..., : n // 2]), dim=-1)
    return torch.cat((x1 * cos + x_rot * sin, x2), dim=-1)

# ------------------------------
# GPT MLP
# ------------------------------
class LLaMAMLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )

    def forward(self, x):
        return self.net(x)

# ------------------------------
# Attention
# ------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.n_head = config.n_head
        self.head_size = config.head_size
        self.attn = nn.Linear(config.n_embd, 3 * self.n_head * self.head_size, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x, rope, max_seq_length, mask=None, input_pos=None, kv_cache=None):
        B, T, C = x.shape
        qkv = self.attn(x).view(B, T, 3, self.n_head, self.head_size)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        cos, sin = rope
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # KV-cache dynamic handling
        if kv_cache is not None and input_pos is not None:
            cache_k, cache_v = kv_cache
            pos = input_pos[0].item() if input_pos.numel() == 1 else 0
            end_pos = min(pos + k.shape[1], max_seq_length)
            len_to_write = end_pos - pos
            if cache_k.shape[2] < end_pos:
                # dynamically expand cache
                new_shape = (cache_k.shape[0], cache_k.shape[1], end_pos, cache_k.shape[3])
                cache_k = torch.cat([cache_k, torch.zeros(new_shape[0], new_shape[1], end_pos - cache_k.shape[2], new_shape[3], device=cache_k.device, dtype=cache_k.dtype)], dim=2)
                cache_v = torch.cat([cache_v, torch.zeros_like(cache_k)], dim=2)
            cache_k[:, :, pos:end_pos, :] = k.transpose(1,2)[:, :, :len_to_write, :]
            cache_v[:, :, pos:end_pos, :] = v.transpose(1,2)[:, :, :len_to_write, :]
            kv_cache = (cache_k, cache_v)
            k = cache_k[:, :, :end_pos, :].transpose(1,2)
            v = cache_v[:, :, :end_pos, :].transpose(1,2)
            T = k.shape[1]

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, scale=1.0 / math.sqrt(self.head_size), is_causal=mask is None)
        y = y.transpose(1,2).reshape(B, T, C)
        y = self.proj(y)
        return y, kv_cache

# ------------------------------
# Transformer Block
# ------------------------------
class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = LLaMAMLP(config)
        self.parallel_residual = config.parallel_residual

    def forward(self, x, rope, max_seq_length, mask=None, input_pos=None, kv_cache=None):
        n1 = self.norm_1(x)
        h, kv_cache = self.attn(n1, rope, max_seq_length, mask, input_pos, kv_cache)
        if self.parallel_residual:
            x = x + h + self.mlp(n1)
        else:
            x = x + h
            x = x + self.mlp(self.norm_2(x))
        return x, kv_cache

# ------------------------------
# GPT Model
# ------------------------------
class GPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": config.norm_class(config.n_embd, eps=config.norm_eps),
        })
        self.rope_cache = None
        self.mask_cache = None
        self.kv_caches = []

    def reset_cache(self):
        self.kv_caches.clear()
        self.rope_cache = None
        self.mask_cache = None

    def forward(self, idx, emb_diff=None, visual_features=None, max_seq_length=None, input_pos=None):
        B, T = idx.shape
        block_size = self.config.block_size
        use_kv_cache = input_pos is not None
        if max_seq_length is None:
            max_seq_length = block_size
        assert T <= block_size, f"Input length {T} > block_size {block_size}"

        if self.rope_cache is None:
            n_elem = int(self.config.rotary_percentage * self.config.head_size)
            self.rope_cache = build_rope_cache(seq_len=block_size, n_elem=n_elem, dtype=idx.dtype, device=idx.device)
        cos, sin = self.rope_cache

        mask = None
        if use_kv_cache and self.mask_cache is None:
            self.mask_cache = torch.tril(torch.ones((block_size, block_size), device=idx.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        if use_kv_cache:
            cos = cos.index_select(0, input_pos)
            sin = sin.index_select(0, input_pos)
            mask = self.mask_cache.index_select(2, input_pos)
            mask = mask[:, :, :, :max_seq_length]
        else:
            cos = cos[:T]
            sin = sin[:T]

        x = self.transformer.wte(idx)
        if not use_kv_cache:
            for block in self.transformer.h:
                x, _ = block(x, (cos, sin), max_seq_length)
        else:
            if not self.kv_caches:
                self.kv_caches = [(
                    torch.zeros(B, self.config.n_head, max_seq_length, self.config.head_size, device=x.device, dtype=x.dtype),
                    torch.zeros(B, self.config.n_head, max_seq_length, self.config.head_size, device=x.device, dtype=x.dtype),
                ) for _ in range(self.config.n_layer)]
            for i, block in enumerate(self.transformer.h):
                x, self.kv_caches[i] = block(x, (cos, sin), max_seq_length, mask, input_pos, self.kv_caches[i])
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits, self.kv_caches

# ------------------------------
# GENERATE function
# ------------------------------
@torch.no_grad()
def generate(model, idx, emb_diff=None, visual_features=None, max_returned_tokens=128, temperature=1.0, top_k=1, eos_id=2):
    output = idx
    for _ in range(max_returned_tokens):
        input_pos = torch.tensor([output.shape[1]-1], device=output.device)
        logits, _ = model(output, emb_diff, visual_features, max_seq_length=model.config.block_size, input_pos=input_pos)
        logits = logits[:, -1, :] / temperature
        if top_k > 0:
            v, ix = torch.topk(logits, top_k)
            probs = torch.zeros_like(logits).scatter_(1, ix, torch.softmax(v, dim=-1))
        else:
            probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        output = torch.cat([output, next_token], dim=1)
        if next_token.item() == eos_id:
            break
    return output[:, idx.shape[1]:]

# ------------------------------
# TOKENIZER
# ------------------------------
class TokenizerClass:
    def __init__(self, path):
        path = Path(path)
        if (path / "tokenizer.model").exists():
            from sentencepiece import SentencePieceProcessor
            self.processor = SentencePieceProcessor(model_file=str(path / "tokenizer.model"))
            self.eos_id = self.processor.eos_id()
        elif (path / "tokenizer.json").exists():
            from tokenizers import Tokenizer as HFTokenizer
            self.processor = HFTokenizer.from_file(str(path / "tokenizer.json"))
            self.eos_id = 2
        else:
            raise RuntimeError("Tokenizer not found")
    def decode(self, tokens):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return self.processor.decode(tokens)

# ------------------------------
# MAIN
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    torch.set_float32_matmul_precision("high")

    # Paths
    BASE_MODEL_DIR = Path(r"E:\ASR\TinyLlama\open_llama_3b")
    ADAPTER_PATH = Path(r"E:\ASR\runs\lipger_epoch2\adapter_model.safetensors")
    EXP_DIR = Path(r"E:\ASR\runs\lipger_epoch2")
    PRED_DIR = EXP_DIR / "predictions"
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    # Load config + model
    with open(BASE_MODEL_DIR / "lit_config.json") as f:
        config = Config(**json.load(f))
    model = GPT(config)
    state = load_file(str(ADAPTER_PATH))
    model.load_state_dict(state, strict=False)
    model.eval()
    model.to(dtype).to(device)

    tokenizer = TokenizerClass(BASE_MODEL_DIR)

    # Safe data load
    data = torch.load(args.test_data, weights_only=False)

    # Preprocessing
    preprocess = lambda x: (torch.tensor(x, dtype=torch.float32) - 0.421)/0.165

    preds, gts, results = [], [], []

    for i, sample in enumerate(tqdm(data)):
        idx = sample["input_ids_no_response"].to(device)
        if idx.numel() == 0 or idx.shape[-1] == 0:
            idx = torch.tensor([[tokenizer.eos_id]], device=device)
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
        emb_diff = sample["emb_diff"].to(device).to(dtype)
        mouth = torch.tensor(h5py.File(sample["mouthroi"], "r")["video_frames"][:], dtype=torch.float32)
        mouth = preprocess(mouth).unsqueeze(0).unsqueeze(0).to(device).to(dtype)
        gt = sample["ground_truth"].strip()
        model.reset_cache()
        with torch.autocast(device_type="cuda", dtype=dtype):
            out_tokens = generate(model, idx, emb_diff, mouth, max_returned_tokens=min(model.config.block_size, idx.shape[1]+150), temperature=0.2, top_k=1, eos_id=tokenizer.eos_id)
        pred = tokenizer.decode(out_tokens).strip()
        preds.append(pred)
        gts.append(gt)
        results.append({"gt": gt, "pred": pred})

    # WER
    wer_metric = load("wer")
    wer = wer_metric.compute(predictions=preds, references=gts)
    results.append({"WER": wer})

    # Save
    out_path = PRED_DIR / "lipger_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✅ Inference Done. WER: {wer*100:.2f}% | Saved at {out_path}")

if __name__ == "__main__":
    main()
