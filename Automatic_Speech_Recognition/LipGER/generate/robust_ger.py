import json
import sys
import time
import warnings
from pathlib import Path
from typing import Literal, Optional

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy

# ---------------- PATH SETUP ----------------
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lipger.model import GPT, Block, Config
from lipger.tokenizer import Tokenizer
from lipger.utils import (
    check_valid_checkpoint_dir,
    get_default_supported_precision,
    lazy_load
)

# ---------------- GENERATE FUNCTION ----------------
@torch.no_grad()
def generate(
    model: torch.nn.Module,
    emb_diff: torch.Tensor,
    visual_features: torch.Tensor,
    idx: torch.Tensor,
    max_returned_tokens: int,
    max_seq_length: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Generates tokens given a prompt and visual/embedding features.
    Compatible with LipGER.
    """
    T = idx.size(0)
    device, dtype = idx.device, idx.dtype
    empty = torch.empty(max_returned_tokens, dtype=dtype, device=device)
    empty[:T] = idx
    idx = empty
    input_pos = torch.arange(0, T, device=device)
    emb_diff = emb_diff.unsqueeze(0)  # add batch dimension

    for _ in range(max_returned_tokens - T):
        x = idx.index_select(0, input_pos).view(1, -1)
        logits, _ = model(x, emb_diff, visual_features, max_seq_length, input_pos)
        logits = logits[0, -1] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(logits < v[[-1]], -float("Inf"), logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1).to(dtype=dtype)

        input_pos = input_pos[-1:] + 1
        idx = idx.index_copy(0, input_pos, idx_next)

        if eos_id is not None and idx_next.item() == eos_id:
            return idx[:input_pos]

    return idx

# ---------------- MAIN FUNCTION ----------------
def main(
    prompt: str = "Hello, my name is",
    *,
    num_samples: int = 1,
    max_new_tokens: int = 50,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8", "gptq.int4"]] = None,
    strategy: str = "auto",
    devices: int = 1,
    precision: Optional[str] = None,
) -> None:
    """
    Generates text from a pretrained GPT model with Lightning Fabric.
    """
    precision = precision or get_default_supported_precision(training=False)
    
    if strategy == "fsdp":
        strategy = FSDPStrategy(auto_wrap_policy=lambda m: isinstance(m, Block), cpu_offload=False)

    fabric = L.Fabric(devices=devices, precision=precision, strategy=strategy)
    fabric.launch()

    check_valid_checkpoint_dir(checkpoint_dir)
    with open(checkpoint_dir / "lit_config.json") as fp:
        config = Config(**json.load(fp))

    model_file = "lit_model.pth" if quantize != "gptq.int4" else "lit_model_gptq.4bit.pth"
    checkpoint_path = checkpoint_dir / model_file
    if quantize == "gptq.int4" and not checkpoint_path.is_file():
        raise ValueError("Please run `python quantize/gptq.py` first")

    fabric.print(f"Loading model {checkpoint_path} with config: {config.__dict__}", file=sys.stderr)
    
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        model = GPT(config)
    fabric.print(f"Model instantiated in {time.perf_counter()-t0:.2f}s", file=sys.stderr)

    t0 = time.perf_counter()
    with lazy_load(checkpoint_path) as ckpt:
        model.load_state_dict(ckpt.get("model", ckpt), strict=quantize is None)
    fabric.print(f"Model weights loaded in {time.perf_counter()-t0:.2f}s", file=sys.stderr)

    model.eval()
    model = fabric.setup_module(model)

    tokenizer = Tokenizer(checkpoint_dir)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens
    assert max_returned_tokens <= model.config.block_size

    L.seed_everything(1234)
    for i in range(num_samples):
        t0 = time.perf_counter()
        y = generate(
            model=model,
            emb_diff=encoded,  # placeholder embedding
            visual_features=torch.zeros(1, 1, 88, 88, device=fabric.device),  # placeholder visual
            idx=encoded,
            max_returned_tokens=max_returned_tokens,
            max_seq_length=max_returned_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=tokenizer.eos_id
        )
        t = time.perf_counter() - t0

        if hasattr(model, "reset_cache"):
            model.reset_cache()
        decoded = tokenizer.decode(y)
        fabric.print(f"Sample {i+1}: {decoded}")
        tokens_generated = y.size(0) - prompt_length
        fabric.print(f"Time: {t:.2f}s, Speed: {tokens_generated/t:.2f} tokens/sec", file=sys.stderr)

    if fabric.device.type == "cuda":
        fabric.print(f"Max GPU memory used: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", file=sys.stderr)

# ---------------- ENTRY POINT ----------------
if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        "ignore",
        message="ComplexHalf support is experimental and many operators don't support it yet",
    )
    CLI(main, allow_abbrev=False)
