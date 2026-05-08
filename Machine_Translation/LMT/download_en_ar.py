from datasets import load_dataset

print("Downloading EN-AR dataset...")

dataset = load_dataset("ymoslem/CoVoST2-EN-AR-Text")

dataset.save_to_disk("data/en_ar_full")

print("✅ EN-AR dataset downloaded & saved")
