from datasets import load_dataset

print("Downloading LMT-60-sft-data (streaming)...")

dataset = load_dataset(
    "LMT-60/LMT-60-sft-data",
    split="train",
    streaming=True
)

print("Dataset stream ready")
