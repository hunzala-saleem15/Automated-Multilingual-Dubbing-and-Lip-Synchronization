input_file = "data/train.jsonl"
output_file = "data/train_small.jsonl"

MAX_LINES = 20000  # sirf 20k samples

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

with open(output_file, "w", encoding="utf-8") as f:
    for line in lines[:MAX_LINES]:
        f.write(line)

print("✅ train_small.jsonl created with", MAX_LINES, "samples")
