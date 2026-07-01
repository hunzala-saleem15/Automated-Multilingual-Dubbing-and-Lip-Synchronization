# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Type, Union
import torch
import lipger.model
from lipger.utils import find_multiple

@dataclass
class Config:
    name: str = ""
    hf_config: dict = field(default_factory=dict)
    scale_embeddings: bool = False
    block_size: int = 4096
    vocab_size: int = 50254
    padding_multiple: int = 512
    padded_vocab_size: int = None
    n_layer: int = 16
    n_head: int = 32
    head_size: int = None
    n_embd: int = 4096
    rotary_percentage: float = 0.25
    parallel_residual: bool = True
    bias: bool = True
    lm_head_bias: bool = False
    n_query_groups: int = None
    shared_attention_norm: bool = False
    _norm_class: str = "LayerNorm"
    norm_eps: float = 1e-5
    _mlp_class: str = "GptNeoxMLP"
    gelu_approximate: str = "none"
    intermediate_size: int = None
    rope_condense_ratio: int = 1
    rope_base: int = 10000
    n_expert: int = 0
    n_expert_per_token: int = 0

    def __post_init__(self):
        if not self.name:
            self.name = self.hf_config.get("name", self.name)

        if self.head_size is None:
            assert self.n_embd % self.n_head == 0
            self.head_size = self.n_embd // self.n_head

        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, self.padding_multiple)
        else:
            self.vocab_size = min(self.vocab_size, self.padded_vocab_size)

        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head

        if self.intermediate_size is None:
            if self._mlp_class == "LLaMAMLP":
                raise ValueError("The config needs to set the `intermediate_size`")
            self.intermediate_size = 4 * self.n_embd

        self.rope_n_elem = int(self.rotary_percentage * self.head_size)

    @property
    def mlp_class(self) -> Type:
        return getattr(lip.model, self._mlp_class)

    @property
    def norm_class(self) -> Type:
        if self._norm_class == "RMSNorm":
            from lipger.rmsnorm import RMSNorm
            return RMSNorm
        return getattr(torch.nn, self._norm_class)

    # -------------------------------
    # Classmethods to create configs
    # -------------------------------
    @classmethod
    def from_name(cls, name: str, **kwargs) -> "Config":
        if name not in name_to_config:
            raise ValueError(f"{name!r} is not a supported config name")
        conf_dict = name_to_config[name].copy()
        conf_dict.update(kwargs)
        return cls(**conf_dict)

    @classmethod
    def from_json(cls, path: Union[str, Path], **kwargs) -> "Config":
        with open(path, encoding="utf-8") as fp:
            json_kwargs = json.load(fp)
        json_kwargs.update(kwargs)
        return cls(**json_kwargs)

    @classmethod
    def from_checkpoint(cls, path: Union[str, Path], **kwargs) -> "Config":
        path = Path(path)
        if (config_path := path / "lit_config.json").is_file():
            return cls.from_json(config_path, **kwargs)
        if path.name in name_to_config:
            return cls.from_name(path.name, **kwargs)
        raise FileNotFoundError(f"For {str(path)!r} neither 'lit_config.json' nor matching config exists.")

# ==========================
# Open LLaMA Configs
# ==========================
open_LLaMA = [
    dict(
        name="open_llama_3b",
        hf_config=dict(org="openlm-research", name="open_llama_3b"),
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=26,
        n_embd=3200,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-6,
        _mlp_class="LLaMAMLP",
        intermediate_size=8640,
    ),
    dict(
        name="open_llama_7b",
        hf_config=dict(org="openlm-research", name="open_llama_7b"),
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-6,
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
    ),
    dict(
        name="open_llama_13b",
        hf_config=dict(org="openlm-research", name="open_llama_13b"),
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-6,
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
    ),
]

configs = []
configs.extend(open_LLaMA)
name_to_config = {c["hf_config"]["name"]: c for c in configs}
