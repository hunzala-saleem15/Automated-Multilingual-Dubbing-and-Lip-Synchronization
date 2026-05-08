import json
import random

data = []
with open("data/train_en_zh.jsonl", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

random.shuffle(data)

test = data[:500]        # small test
train = data[500:]

with open("data/test2.en", "w", encoding="utf-8") as e, \
     open("data/test2.zh", "w", encoding="utf-8") as z:
    for x in test:
        e.write(x["src"].strip() + "\n")
        z.write(x["tgt"].strip() + "\n")

print("✅ Test set created:", len(test))
