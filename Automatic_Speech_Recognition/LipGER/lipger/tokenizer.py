import json
from pathlib import Path
from typing import Optional

import torch


class Tokenizer:
    def __init__(self, checkpoint_dir: Path) -> None:
        self.use_bos = self.check_if_bos_token_used(checkpoint_dir)
        self.bos_id = None
        self.eos_id = None

        # ---------------- SentencePiece (priority) ----------------
        if (vocabulary_path := checkpoint_dir / "tokenizer.model").is_file():
            from sentencepiece import SentencePieceProcessor

            self.processor = SentencePieceProcessor(model_file=str(vocabulary_path))
            self.backend = "sentencepiece"
            self.bos_id = self.processor.bos_id()
            self.eos_id = self.processor.eos_id()

        # ---------------- HuggingFace Tokenizer ----------------
        elif (vocabulary_path := checkpoint_dir / "tokenizer.json").is_file():
            from tokenizers import Tokenizer as HFTokenizer

            self.processor = HFTokenizer.from_file(str(vocabulary_path))
            self.backend = "huggingface"

            # tokenizer_config.json
            if (cfg_path := checkpoint_dir / "tokenizer_config.json").is_file():
                with open(cfg_path, encoding="utf-8") as fp:
                    config = json.load(fp)

                bos_token = config.get("bos_token")
                if isinstance(bos_token, dict):
                    bos_token = bos_token.get("content")
                if isinstance(bos_token, str):
                    self.bos_id = self.token_to_id(bos_token)

                eos_token = config.get("eos_token")
                if isinstance(eos_token, dict):
                    eos_token = eos_token.get("content")
                if isinstance(eos_token, str):
                    self.eos_id = self.token_to_id(eos_token)

            # generation_config.json fallback
            if (gen_path := checkpoint_dir / "generation_config.json").is_file():
                with open(gen_path, encoding="utf-8") as fp:
                    gen_cfg = json.load(fp)
                if self.bos_id is None:
                    self.bos_id = gen_cfg.get("bos_token_id")
                if self.eos_id is None:
                    self.eos_id = gen_cfg.get("eos_token_id")

        else:
            raise NotImplementedError("No tokenizer.model or tokenizer.json found")

    # ------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        if self.backend == "huggingface":
            return self.processor.get_vocab_size(with_added_tokens=False)
        if self.backend == "sentencepiece":
            return self.processor.vocab_size()
        raise RuntimeError

    def token_to_id(self, token: str) -> int:
        if not isinstance(token, str):
            raise TypeError(f"token must be str, got {type(token)}")

        if self.backend == "huggingface":
            id_ = self.processor.token_to_id(token)
        elif self.backend == "sentencepiece":
            id_ = self.processor.piece_to_id(token)
        else:
            raise RuntimeError

        if id_ is None:
            raise ValueError(f"token {token!r} not found in tokenizer")
        return id_

    # ------------------------------------------------------------

    def check_if_bos_token_used(self, checkpoint_dir: Path) -> bool:
        cfg = checkpoint_dir / "tokenizer_config.json"
        if not cfg.is_file():
            return False

        with open(cfg, encoding="utf-8") as fp:
            config = json.load(fp)

        if any(config.get(k, False) for k in ("add_bos_token", "add_prefix_space")):
            return True

        return (
            config.get("add_bos_token") is None
            and config.get("tokenizer_class") == "LlamaTokenizer"
        )

    # ------------------------------------------------------------

    def encode(
        self,
        string: str,
        device: Optional[torch.device] = None,
        bos: Optional[bool] = None,
        eos: bool = False,
        max_length: int = -1,
    ) -> torch.Tensor:
        if self.backend == "huggingface":
            tokens = self.processor.encode(string).ids
        elif self.backend == "sentencepiece":
            tokens = self.processor.encode(string)
        else:
            raise RuntimeError

        if bos or self.use_bos:
            if self.bos_id is None:
                raise RuntimeError("BOS token requested but not defined")
            tokens = [self.bos_id] + tokens

        if eos and self.eos_id is not None:
            tokens = tokens + [self.eos_id]

        if max_length > 0:
            tokens = tokens[:max_length]

        return torch.tensor(tokens, dtype=torch.long, device=device)

    # ------------------------------------------------------------

    def decode(self, tensor: torch.Tensor) -> str:
        tokens = tensor.tolist() if tensor.ndim > 0 else [tensor.item()]

        if self.backend == "huggingface":
            return self.processor.decode(tokens, skip_special_tokens=True)
        elif self.backend == "sentencepiece":
            return self.processor.decode(tokens)
        else:
            raise RuntimeError
