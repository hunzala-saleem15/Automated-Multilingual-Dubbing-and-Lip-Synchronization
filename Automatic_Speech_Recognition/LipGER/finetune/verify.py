import os
import torch
import torchaudio
import json
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

# =====================================================
# CONFIG
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
JSON_FILE = r"E:\ASR\facestar_whisper\facestar_full_test_whisper_fixed_clean_backup.json"

print("Using device:", DEVICE)

# =====================================================
# TEXT NORMALIZATION
# =====================================================
transform = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    Strip()
])

# =====================================================
# LOAD WHISPER-LARGE-V2
# =====================================================
print("\n[INFO] Loading Whisper-Large-v2 model...")
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2").to(DEVICE)
model.eval()

# =====================================================
# LOAD JSON DATA
# =====================================================
with open(JSON_FILE, "r", encoding="utf-8") as f:
    data_list = json.load(f)

# =====================================================
# FUNCTION TO PROCESS ONE SPEAKER
# =====================================================
def process_speaker(audio_dir):
    results = []
    total_wer = 0
    count = 0

    # Normalize paths to avoid mismatch
    audio_dir_norm = os.path.normpath(audio_dir)

    # Filter items in JSON that belong to this audio_dir
    items = [item for item in data_list if os.path.normpath(os.path.dirname(item["Clean_Wav"])) == audio_dir_norm]

    for item in items:
        audio_path = item["Clean_Wav"]
        reference = item["Caption"]

        if not os.path.exists(audio_path):
            print(f"[WARNING] Audio file not found: {audio_path}")
            continue

        # ---- LOAD AUDIO ----
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.squeeze(0)

        # Resample if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
            sr = 16000

        # ---- GENERATE WHISPER TRANSCRIPTION ----
        inputs = processor(
            waveform,
            sampling_rate=sr,
            return_tensors="pt"
        )
        input_features = inputs.input_features.to(DEVICE)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                language="en",
                task="transcribe"
            )

        transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # ---- CALCULATE WER ----
        ref_norm = transform(reference)
        hyp_norm = transform(transcript)
        file_wer = wer(ref_norm, hyp_norm)

        print(f"\nFile: {os.path.basename(audio_path)}")
        print(f"Whisper Output: {transcript}")
        print(f"Reference: {reference}")
        print(f"WER: {file_wer*100:.2f}%")

        results.append({
            "file": os.path.basename(audio_path),
            "transcript": transcript,
            "reference": reference,
            "wer": file_wer
        })
        total_wer += file_wer
        count += 1

    # =====================================================
    # AVERAGE WER
    # =====================================================
    if count > 0:
        avg_wer = (total_wer / count) * 100
        print("\n==============================")
        print(f"Processed files: {count}")
        print(f"Average Whisper WER: {avg_wer:.2f}%")
        print("==============================\n")
    else:
        avg_wer = None
        print("\nNo valid audio files found. Cannot calculate average WER.\n")

    return results, avg_wer

# =====================================================
# PROCESS BOTH SPEAKERS
# =====================================================
SPEAKERS_DIRS = {
    "Male": r"E:\ASR\facestar\male_speaker\testset",
    "Female": r"E:\ASR\facestar\female_speaker\testset"
}

all_results = {}
speaker_avg_wer = {}

for spk, spk_dir in SPEAKERS_DIRS.items():
    print(f"\n========== Processing {spk} Speaker ==========")
    results, avg_wer = process_speaker(spk_dir)
    all_results[spk] = results
    speaker_avg_wer[spk] = avg_wer

# =====================================================
# COMBINED AVERAGE WER (Male + Female)
# =====================================================
valid_wers = [v for v in speaker_avg_wer.values() if v is not None]
if len(valid_wers) > 0:
    combined_avg = sum(valid_wers) / len(valid_wers)
    print(f"\nCombined Average WER (Male + Female): {combined_avg:.2f}%")
else:
    print("\nNo valid files found for any speaker. Cannot calculate combined WER.")