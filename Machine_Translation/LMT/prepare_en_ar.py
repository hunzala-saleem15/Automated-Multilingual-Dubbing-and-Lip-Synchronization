from datasets import load_from_disk
import json

dataset = load_from_disk("data/en_ar_full")
train_data = dataset["train"]

with open("data/train_en_ar.jsonl", "w", encoding="utf-8") as f:
    for item in train_data:
        record = {
            "src": item["text_en"],   # English
            "tgt": item["text_ar"]    # Arabic
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("✅ train_en_ar.jsonl created successfully")
