import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model

MODEL_NAME = "NiuTrans/LMT-60-1.7B"

def main():

    print("Loading OPUS100 EN-HI dataset...")
    dataset = load_dataset("opus100", "en-hi")

    train_data = dataset["train"].shuffle(seed=42).select(range(10000))
    val_data = dataset["validation"].select(range(500))

    print("Train samples:", len(train_data))
    print("Validation samples:", len(val_data))

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        padding_side="left",
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    def build_prompt(example):
        return {
            "text": f"Translate English to Hindi:\nEnglish: {example['translation']['en']}\nHindi:"
        }

    train_data = train_data.map(build_prompt, remove_columns=train_data.column_names)
    val_data = val_data.map(build_prompt, remove_columns=val_data.column_names)

    def tokenize_fn(example):
        tokens = tokenizer(
            example["text"],
            truncation=True,
            max_length=128,
            padding="max_length"
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    train_tokenized = train_data.map(tokenize_fn)
    val_tokenized = val_data.map(tokenize_fn)

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float32,   # ✅ Windows safe
        trust_remote_code=True
    )

    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="./lmt_lora_en_hi",
        overwrite_output_dir=True,

        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,

        max_steps=1200,                 # 🔥 FAST
        learning_rate=2e-4,             # 🔥 LoRA LR

        warmup_steps=100,
        lr_scheduler_type="cosine",

        fp16=False,
        bf16=False,

        logging_steps=25,
        save_steps=1200,
        save_total_limit=1,

        dataloader_num_workers=0,
        dataloader_pin_memory=True,

        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer
    )

    print("Training started (LoRA FAST MODE)...")
    trainer.train()
    print("✅ Training finished successfully")

if __name__ == "__main__":
    main()
