from datasets import load_from_disk

# Load full EN-AR dataset
dataset = load_from_disk("data/en_ar_full")

# Take first 50 samples
samples = dataset["train"][:50]

with open("data/test1.en", "w", encoding="utf-8") as f_en, \
     open("data/test1.ar", "w", encoding="utf-8") as f_ar:

    for en, ar in zip(samples["text_en"], samples["text_ar"]):
        f_en.write(en.strip() + "\n")
        f_ar.write(ar.strip() + "\n")

print("✅ test1.en and test1.ar created (50 sentences)")
