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
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        padding_side="left"
    )

    model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16
)


    # ============================
    # Dataset Loading
    # ============================

    dataset = load_dataset(
        "json",
        # OLD DATASET (EN → ES) 
        # data_files="../data/train_small.jsonl"

        # NEW DATASET (EN → AR) 
        data_files="data/train_en_ar_small.jsonl"
    )

    # ============================
    # Tokenization
    # ============================

    # OLD TOKENIZER (EN → ES) 
    # def tokenize(example):
    #     text = (
    #         f"{example['instruction']}\n"
    #         f"{example['input']}\n"
    #         f"{example['output']}"
    #     )

    # NEW TOKENIZER (EN → AR) ✅
    def tokenize(example):
        text = f"English: {example['src']}\nArabic: {example['tgt']}"

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
        num_proc=8
    )

    # ============================
    # Training Arguments
    # ============================

    training_args = TrainingArguments(
    output_dir="./lmt_finetuned_en_ar_fast",

    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,

    num_train_epochs=2,
    max_steps=5000,

    learning_rate=2e-5,

    fp16=False,
    bf16=True,

    logging_steps=50,
    save_steps=1000,
    save_total_limit=1,

    report_to="none"
)



    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        tokenizer=tokenizer
    )

    trainer.train()

if __name__ == "__main__":
    main()
