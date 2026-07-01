import os
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ======================
# Paths
# ======================
MALE_AUDIO_DIR   = r"E:\ASR\facestar\male_speaker\testset"
MALE_REF_DIR     = r"E:\ASR\facestar\male_speaker\refs"
FEMALE_AUDIO_DIR = r"E:\ASR\facestar\female_speaker\testset"
FEMALE_REF_DIR   = r"E:\ASR\facestar\female_speaker\refs"

# ======================
# Set HF cache to E: drive
# ======================
os.environ["HF_HOME"] = r"E:\ASR\.cache\huggingface"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # disable symlink warning on Windows

# ======================
# Device
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ======================
# Load Whisper Processor and Model
# ======================
print("[INFO] Loading Whisper-Large-v2 model...")
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v2",
    torch_dtype=torch.float16
).to(DEVICE)
model.eval()

# ======================
# Function to generate references
# ======================
def generate_refs(audio_dir, ref_dir):
    os.makedirs(ref_dir, exist_ok=True)

    for audio_file in os.listdir(audio_dir):
        if not audio_file.lower().endswith(".wav"):
            continue

        audio_path = os.path.join(audio_dir, audio_file)
        ref_path   = os.path.join(ref_dir, audio_file.replace(".wav", ".txt"))

        if os.path.exists(ref_path):
            print(f"[SKIPPED] Already exists: {audio_file}")
            continue

        print(f"[PROCESSING] {audio_file}")

        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.squeeze(0)

        # Resample if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)

        # Prepare input features
        inputs = processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt"
        )
        input_features = inputs.input_features.to(DEVICE, dtype=torch.float16)  # <-- FIX float16

        # Generate transcript
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                do_sample=False,
                num_beams=1,
                language="en",
                task="transcribe"
            )

        transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # Save transcript
        with open(ref_path, "w", encoding="utf-8") as f:
            f.write(transcript)

        print(f"[SAVED] {ref_path}")

# ======================
# Generate Male References
# ======================
print("\n✅ Generating Male References...")
generate_refs(MALE_AUDIO_DIR, MALE_REF_DIR)

# ======================
# Generate Female References
# ======================
print("\n✅ Generating Female References...")
generate_refs(FEMALE_AUDIO_DIR, FEMALE_REF_DIR)

print("\n🎉 All references generated successfully!")