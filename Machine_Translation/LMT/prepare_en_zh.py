from datasets import load_dataset
import json
import os

os.makedirs("data", exist_ok=True)

dataset = load_dataset(
    "json",
    data_files="hf://datasets/NiuTrans/LMT-60-sft-data/en-zh.jsonl",
    split="train"
)

out_path = "data/train_en_zh.jsonl"
count = 0

with open(out_path, "w", encoding="utf-8") as f:
    for item in dataset:
        trans = item.get("translation", {})

        src = trans.get("en", "").strip()
        tgt = trans.get("zh", "").strip()

        if src and tgt:
            json.dump({"src": src, "tgt": tgt}, f, ensure_ascii=False)
            f.write("\n")
            count += 1

print(f"✅ EN→ZH dataset prepared with {count} samples")
