"""Utility functions for training, preprocessing, and inference."""

import os
import sys
import warnings
from contextlib import contextmanager
from functools import partial
from io import BytesIO
from pathlib import Path
from types import MethodType
from typing import Dict, List, Mapping, Optional, Type, TypeVar, Union, Any

import random
import torch
import torch.nn as nn
from torch.serialization import normalize_storage_type

from lightning.fabric.loggers import CSVLogger

# ----------------------------
# Helpers
# ----------------------------
def find_multiple(n: int, k: int) -> int:
    assert k > 0
    if n % k == 0:
        return n
    return n + k - (n % k)

def num_parameters(module: nn.Module, requires_grad: Optional[bool] = None) -> int:
    return sum(p.numel() for p in module.parameters() if requires_grad is None or p.requires_grad == requires_grad)

# ----------------------------
# Image preprocessing / augmentation
# ----------------------------
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class Normalize:
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std

class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        c, h, w = x.shape
        th, tw = self.size
        if h == th and w == tw:
            return x
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return x[:, i:i+th, j:j+tw]

class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        c, h, w = x.shape
        th, tw = self.size
        i = (h - th) // 2
        j = (w - tw) // 2
        return x[:, i:i+th, j:j+tw]

# ----------------------------
# Quantization context manager
# ----------------------------
@contextmanager
def quantization(mode: Optional[str] = None):
    if mode is None:
        yield
        return

    if mode.startswith("bnb"):
        import quantize.bnb as bnb

    if mode == "bnb.int8":
        quantized_linear_cls = bnb.InferenceLinear8bitLt
    elif mode == "bnb.fp4":
        class QuantizedLinear(bnb.Linear4bit):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, quant_type="fp4", compress_statistics=False, **kwargs)
        quantized_linear_cls = QuantizedLinear
    elif mode == "bnb.fp4-dq":
        class QuantizedLinear(bnb.Linear4bit):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, quant_type="fp4", compress_statistics=True, **kwargs)
        quantized_linear_cls = QuantizedLinear
    elif mode == "bnb.nf4":
        class QuantizedLinear(bnb.Linear4bit):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, quant_type="nf4", compress_statistics=False, **kwargs)
        quantized_linear_cls = QuantizedLinear
    elif mode == "bnb.nf4-dq":
        class QuantizedLinear(bnb.Linear4bit):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, quant_type="nf4", compress_statistics=True, **kwargs)
        quantized_linear_cls = QuantizedLinear
    elif mode == "gptq.int4":
        from quantize.gptq import ColBlockQuantizedLinear
        class QuantizedLinear(ColBlockQuantizedLinear):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, bits=4, tile_cols=-1, **kwargs)
        quantized_linear_cls = QuantizedLinear
    else:
        raise ValueError(f"Unknown quantization mode: {mode}")

    torch_linear_cls = torch.nn.Linear
    torch.nn.Linear = quantized_linear_cls
    yield
    torch.nn.Linear = torch_linear_cls

# ----------------------------
# HuggingFace checkpoint loader
# ----------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer

class LazyLoadHF:
    def __init__(self, model_path: Union[str, Path]):
        self.model_path = str(model_path)
        self.model = None
        self.tokenizer = None

    def __enter__(self):
        print(f"[INFO] Loading model from {self.model_path} ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, local_files_only=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True
        )
        print("[INFO] Model loaded successfully!")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        print("[INFO] Model unloaded from memory.")

# ----------------------------
# Checkpoint validation
# ----------------------------
def check_valid_checkpoint_dir(checkpoint_dir: Path) -> None:
    files = {
        "pytorch_model.bin": (checkpoint_dir / "pytorch_model.bin").is_file(),
        "config.json": (checkpoint_dir / "config.json").is_file(),
        "tokenizer.json OR tokenizer.model": (checkpoint_dir / "tokenizer.json").is_file() or (checkpoint_dir / "tokenizer.model").is_file(),
        "tokenizer_config.json": (checkpoint_dir / "tokenizer_config.json").is_file(),
    }
    if checkpoint_dir.is_dir() and all(files.values()):
        return
    problem = "missing files: " + ", ".join(f for f, exists in files.items() if not exists) if checkpoint_dir.is_dir() else "not a directory"
    print(f"--checkpoint_dir {checkpoint_dir} {problem}", file=sys.stderr)
    raise SystemExit(1)

# ----------------------------
# Correct chunked cross-entropy
# ----------------------------
def chunked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, chunk_size: int = 128, ignore_index: int = -1) -> torch.Tensor:
    """
    Compute cross-entropy loss in chunks for causal LM (handles shifted targets).
    """
    # Align sequence length
    if logits.size(1) != targets.size(1):
        logits = logits[:, :targets.size(1), :]

    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)

    if chunk_size <= 0 or chunk_size >= B*T:
        loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat, ignore_index=ignore_index, reduction="none")
    else:
        loss_chunks = []
        for i in range(0, B*T, chunk_size):
            lc = logits_flat[i:i+chunk_size]
            tc = targets_flat[i:i+chunk_size]
            loss_chunks.append(torch.nn.functional.cross_entropy(lc, tc, ignore_index=ignore_index, reduction="none"))
        loss = torch.cat(loss_chunks)

    return loss.view(B, T).mean()

# ----------------------------
# Checkpoint loading
# ----------------------------
def load_checkpoint(fabric, model, checkpoint_path: Path, strict: bool = True) -> None:
    from transformers import AutoModelForCausalLM
    hf_model = AutoModelForCausalLM.from_pretrained(checkpoint_path, local_files_only=True)
    model.load_state_dict(hf_model.state_dict(), strict=strict)

# ----------------------------
# CSVLogger step merge
# ----------------------------
T = TypeVar("T")
def step_csv_logger(*args: Any, cls: Type[T] = CSVLogger, **kwargs: Any) -> T:
    logger = cls(*args, **kwargs)
    def merge_by(dicts, key):
        from collections import defaultdict
        out = defaultdict(dict)
        for d in dicts:
            if key in d:
                out[d[key]].update(d)
        return [v for _, v in sorted(out.items())]
    def save(self) -> None:
        import csv
        if not self.metrics: return
        metrics = merge_by(self.metrics, "step")
        keys = sorted({k for m in metrics for k in m})
        with self._fs.open(self.metrics_file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(metrics)
    logger.experiment.save = MethodType(save, logger.experiment)
    return logger
