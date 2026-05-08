import json

inp = "data/train_en_zh_small.jsonl"

src_out = open("data/train.zh.src", "w", encoding="utf-8")
tgt_out = open("data/train.zh.tgt", "w", encoding="utf-8")

with open(inp, encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        src_out.write(item["src"] + "\n")
        tgt_out.write(item["tgt"] + "\n")

src_out.close()
tgt_out.close()

print("✅ Training files ready (EN → ZH)")
