import json

input_file = "data/train_en_zh.jsonl"
output_file = "data/train_en_zh_prompt.jsonl"

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:

    for line in fin:
        obj = json.loads(line)
        src = obj["src"].strip()
        tgt = obj["tgt"].strip()

        prompt = f"en: {src}\nzh: {tgt}"
        fout.write(json.dumps({"text": prompt}, ensure_ascii=False) + "\n")

print("✅ train_en_zh_prompt.jsonl created successfully")
