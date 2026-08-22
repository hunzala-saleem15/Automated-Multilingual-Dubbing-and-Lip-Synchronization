from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

import torch


class MTService:


    def __init__(self):

        path = (
            "app/model_weights/nllb"
        )


        self.tokenizer = AutoTokenizer.from_pretrained(
            path
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            path
        )


        if torch.cuda.is_available():
            self.model.cuda()


        print("NLLB Loaded")


    def translate(
        self,
        text,
        source_lang,
        target_lang
    ):


        self.tokenizer.src_lang = source_lang


        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        )


        if torch.cuda.is_available():
            inputs = {
                k:v.cuda()
                for k,v in inputs.items()
            }


        translated = self.model.generate(
            **inputs,
            forced_bos_token_id=
            self.tokenizer.lang_code_to_id[target_lang]
        )


        return self.tokenizer.decode(
            translated[0],
            skip_special_tokens=True
        )