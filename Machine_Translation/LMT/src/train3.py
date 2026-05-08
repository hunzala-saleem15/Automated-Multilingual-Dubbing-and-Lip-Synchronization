import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)

MODEL_NAME = "NiuTrans/LMT-60-1.7B"

def main():

    # ============================
    # TOKENIZER
    # ============================
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        padding_side="left",
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token

    # ============================
    # MODEL  ❌ FP16 REMOVED
    # ============================
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float32   # ✅ SAFE
    )

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # ============================
    # DATASET
    # ============================
    dataset = load_dataset(
        "json",
        data_files="data/train_en_zh_small.jsonl"
    )

    # ============================
    # TOKENIZATION (CORRECT FORMAT)
    # ============================
    def tokenize(example):
        text = f"English: {example['src']}\nChinese: {example['tgt']}"

        tokens = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=192
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset.map(
        tokenize,
        remove_columns=dataset["train"].column_names,
        num_proc=4
    )

    # ============================
    # TRAINING ARGS  ❌ FP16 OFF
    # ============================
    training_args = TrainingArguments(
        output_dir="./lmt_finetuned_en_zh_balanced",
        overwrite_output_dir=True,

        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,

        num_train_epochs=1,
        max_steps=1200,

        learning_rate=2e-5,
        warmup_steps=120,
        lr_scheduler_type="cosine",

        fp16=False,   # ✅ IMPORTANT
        bf16=False,   # ✅ IMPORTANT

        logging_steps=25,
        save_steps=1200,
        save_total_limit=1,

        report_to="none",
        dataloader_pin_memory=True
    )

    # ============================
    # TRAINER
    # ============================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        tokenizer=tokenizer
    )

    trainer.train()
    print("✅ EN → ZH TRAINING COMPLETED")

if __name__ == "__main__":
    main()
