import json

INPUT = "data/train_en_zh.jsonl"
OUTPUT = "data/train_en_zh_small.jsonl"
MAX = 12000   # safe for your GPU

with open(INPUT, encoding="utf-8") as f:
    lines = f.readlines()

with open(OUTPUT, "w", encoding="utf-8") as f:
    for line in lines[:MAX]:
        f.write(line)

print(f"✅ EN→ZH small dataset created: {MAX} samples")
