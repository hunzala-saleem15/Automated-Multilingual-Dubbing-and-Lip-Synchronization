import json

input_file = "data/train_en_ar.jsonl"
output_file = "data/train_en_ar_small.jsonl"

MAX_LINES = 20000  # fast training ke liye

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

with open(output_file, "w", encoding="utf-8") as f:
    for line in lines[:MAX_LINES]:
        f.write(line)

print("✅ train_en_ar_small.jsonl created with", MAX_LINES, "samples")
