import json
import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from transformers import WhisperModel, WhisperProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Model
# -----------------------------
model_name = "openai/whisper-tiny"  # ya whisper-base
model = WhisperModel.from_pretrained(model_name).to(device)
processor = WhisperProcessor.from_pretrained(model_name)

# -----------------------------
# Whisper embedding function
# -----------------------------
def generate_whisper_embedding(wav_path):
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    waveform, sr = torchaudio.load(wav_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    audio = waveform.squeeze(0).numpy().astype(np.float32)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)

    with torch.no_grad():
        encoder_outputs = model.encoder(inputs).last_hidden_state
        embedding = encoder_outputs.mean(dim=1).cpu().numpy()
    return embedding

# -----------------------------
# Paths
# -----------------------------
json_path = r"E:\ASR\facestar_whisper\facestar_full_train_whisper_fixed_clean.json"

# Backup original JSON
backup_path = json_path.replace(".json", "_backup.json")
if not os.path.exists(backup_path):
    os.rename(json_path, backup_path)

# -----------------------------
# Load JSON
# -----------------------------
with open(backup_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# -----------------------------
# Process each item
# -----------------------------
for item in tqdm(data, desc="Processing JSON"):
    wav_path = item.get("Clean_Wav", None)
    if wav_path is None or not os.path.exists(wav_path):
        print(f"[!] Audio missing: {wav_path}, skipping...")
        continue

    # Whisper embedding path
    whisper_npy = wav_path.replace(".wav", "-aug_new_whisper.npy")
    item["whisper_emb"] = whisper_npy

    # Generate if missing
    if not os.path.exists(whisper_npy):
        emb = generate_whisper_embedding(wav_path)
        np.save(whisper_npy, emb)
        print(f"[+] Generated embedding: {whisper_npy}")

# -----------------------------
# Save updated JSON
# -----------------------------
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)
print(f"[+] Updated JSON saved: {json_path}")