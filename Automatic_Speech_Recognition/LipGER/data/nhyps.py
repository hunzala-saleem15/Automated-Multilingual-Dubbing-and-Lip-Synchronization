# nhyps_whisper_windows_final.py
import sys
import os
import json
import collections
from tqdm import tqdm
import torch
import soundfile as sf
import whisper

# ============================
# 1️⃣ JSON paths
# ============================
test_json_path = r"E:/ASR/facestar_whisper/facestar_full_test_whisper_fixed_clean.json"
train_json_path = r"E:/ASR/facestar_whisper/facestar_full_train_whisper_fixed_clean.json"

output_test_json = r"E:/ASR/facestar_whisper/mciro_test_whisper_tiny.json"
output_train_json = r"E:/ASR/facestar_whisper/mciro_train_whisper_tiny.json"

# ============================
# 2️⃣ Device setup
# ============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Whisper Tiny
model = whisper.load_model("tiny").to(device)
decode_options = {"beam_size": 10}

# ============================
# 3️⃣ Save JSON function
# ============================
def save_json(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Saved intermediate JSON to {filepath}")

# ============================
# 4️⃣ N-best storage
# ============================
d = collections.defaultdict(list)
processed_count = 0
save_interval = 50

# ============================
# 5️⃣ Process JSON function
# ============================
def process_facestar_json(json_path, output_json, use_noisy=True):
    global processed_count

    with open(json_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    for entry in tqdm(data_list, desc=f"Processing {os.path.basename(json_path)}"):
        try:
            path_key = "Noisy_Wav" if use_noisy else "Clean_Wav"
            raw_path = entry[path_key]
            path = os.path.abspath(raw_path).replace("\\", "/")

            if not os.path.exists(path):
                print(f"[WARNING] File not found: {path}")
                continue

            # Load audio and fix dtype
            audio, sr = sf.read(path)
            audio = audio.astype("float32")  # fix float64/double issue
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(device)

            # Decode using the new API
            result = model.decode(mel, **decode_options)

            # ✅ Extract text properly
            if hasattr(result, "text"):
                transcript = result.text
            elif isinstance(result, dict) and "text" in result:
                transcript = result["text"]
            else:
                transcript = str(result)  # fallback

            # N-best placeholder
            d[path] = [transcript] * 10

            processed_count += 1
            if processed_count % save_interval == 0:
                save_json(d, output_json)

        except Exception as e:
            print(f"[ERROR] Processing {raw_path}: {e}")

    # Final save
    save_json(d, output_json)

# ============================
# 6️⃣ Run transcription
# ============================
# Testset
process_facestar_json(test_json_path, output_test_json, use_noisy=True)

# Trainset
process_facestar_json(train_json_path, output_train_json, use_noisy=True)

print("✅ All files processed. Final JSONs saved.")
