import json

INPUT_FILE = "data/train_en_zh_small.jsonl"
OUTPUT_FILE = "data/test_en_zh.en"

with open(INPUT_FILE, encoding="utf-8") as f, open(OUTPUT_FILE, "w", encoding="utf-8") as w:
    for i, line in enumerate(f):
        if i >= 100:   # sirf 100 sentences test ke liye (thesis standard)
            break
        obj = json.loads(line)
        w.write(obj["src"].strip() + "\n")

print("✅ test_en_zh.en created (100 sentences)")
