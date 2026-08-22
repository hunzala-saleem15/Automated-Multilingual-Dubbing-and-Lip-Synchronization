import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class TranslationModel:

    def __init__(
        self,
        model_path=None
    ):

        print("Loading NLLB Translation Model...")

        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "model_weights",
                "nllb"
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Device: {self.device}")
        print(f"Model Path: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            local_files_only=True
        )

        self.model.to(self.device)
        self.model.eval()

        print("NLLB Translation Model loaded successfully.")


    def translate(
        self,
        text,
        source_lang="eng_Latn",
        target_lang="urd_Arab"
    ):

        self.tokenizer.src_lang = source_lang

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        generated_tokens = self.model.generate(
            **encoded,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(target_lang),
            max_new_tokens=512
        )

        translated_text = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0]

        return translated_text


# Load model once
translator = TranslationModel()