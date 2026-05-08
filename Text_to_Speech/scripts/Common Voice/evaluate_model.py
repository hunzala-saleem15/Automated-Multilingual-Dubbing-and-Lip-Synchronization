import os
import torch
import torchaudio
import whisper
import pandas as pd
from jiwer import wer
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# SpeechBrain Import
try:
    from speechbrain.pretrained import EncoderClassifier
except ImportError:
    from speechbrain.inference.speaker import EncoderClassifier

# -----------------------------------------------------------------------------
# 1. PATHS SETUP
# -----------------------------------------------------------------------------
# Training Run Folder
TRAIN_DIR = r"E:\TTS\TTS\recipes\ljspeech\xtts_v2\run\training\GPT_XTTS_v2_English_FineTune-January-18-2026_04+37PM-dbf1a08a"

CONFIG_PATH = os.path.join(TRAIN_DIR, "config.json")
MODEL_PATH = os.path.join(TRAIN_DIR, "best_model.pth") 
VOCAB_PATH = r"E:\TTS\TTS\checkpoints\XTTS_v2\vocab.json"
RESULTS_FILE = "evaluation_results.csv"

# --- SMART REFERENCE AUDIO FINDER (Updated) ---
# Hum in sab jagah check karenge ki wav file kahan chupi hai
POSSIBLE_FOLDERS = [
    r"E:\TTS\TTS\data\XTTS_Ready_Dataset\wavs",  # Processed Data
    r"E:\TTS\TTS\data\wavs",                      # Raw Data
    r"E:\TTS\TTS\data\audio",                     # Raw Audio
    r"E:\TTS\TTS\data"                            # Root Data
]

REF_AUDIO_PATH = None

print("🔍 Searching for Reference Audio...")
for folder in POSSIBLE_FOLDERS:
    if os.path.exists(folder):
        # Folder mein koi bhi .wav file dhoondo
        wav_files = [f for f in os.listdir(folder) if f.endswith(".wav")]
        if wav_files:
            REF_AUDIO_PATH = os.path.join(folder, wav_files[0]) # Pehli file utha lo
            print(f"✅ Found Reference Audio in: {folder}")
            break

# Agar abhi bhi nahi mili, to Error
if not REF_AUDIO_PATH:
    print("\n❌ CRITICAL ERROR: Koi bhi .wav file nahi mili!")
    print("Please manually check ki aapki audio files kahan hain.")
    print("Suggestions:")
    print("1. 'E:\\TTS\\TTS\\data\\XTTS_Ready_Dataset\\wavs' check karein.")
    print("2. Script mein 'REF_AUDIO_PATH' ko manually set karein.")
    exit()

print(f"🎤 Using Reference: {REF_AUDIO_PATH}")

# Test Sentences
test_sentences = [
    "Hello, this is a test to calculate the word error rate.",
    "Artificial intelligence is changing the world rapidly.",
    "Voice cloning allows preserving the identity of the speaker.",
    "The weather today is very nice for a walk."
]

# -----------------------------------------------------------------------------
# 2. LOAD MODELS
# -----------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running Evaluation on: {device}")

# A. Load XTTS
print("⏳ Loading XTTS Model...")
if not os.path.exists(CONFIG_PATH) or not os.path.exists(MODEL_PATH):
    print("❌ Error: Model files (config.json / best_model.pth) nahi mile!")
    exit()

config = XttsConfig()
config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=MODEL_PATH, vocab_path=VOCAB_PATH, use_deepspeed=False)
model.to(device)

# B. Load Whisper
print("⏳ Loading Whisper (ASR)...")
asr_model = whisper.load_model("base", device=device)

# C. Load SpeechBrain
print("⏳ Loading Speaker Encoder...")
spk_encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", 
    run_opts={"device": "cpu"} 
)

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def get_similarity(audio_path1, audio_path2):
    try:
        signal1, fs1 = torchaudio.load(audio_path1)
        signal2, fs2 = torchaudio.load(audio_path2)
        
        # Mono Convert
        if signal1.shape[0] > 1: signal1 = torch.mean(signal1, dim=0, keepdim=True)
        if signal2.shape[0] > 1: signal2 = torch.mean(signal2, dim=0, keepdim=True)

        # Resample 16k
        if fs1 != 16000: signal1 = torchaudio.transforms.Resample(fs1, 16000)(signal1)
        if fs2 != 16000: signal2 = torchaudio.transforms.Resample(fs2, 16000)(signal2)

        signal1, signal2 = signal1.cpu(), signal2.cpu()

        with torch.no_grad():
            emb1 = spk_encoder.encode_batch(signal1)
            emb2 = spk_encoder.encode_batch(signal2)

        vec1, vec2 = emb1.reshape(-1), emb2.reshape(-1)
        return (torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2))).item()

    except Exception as e:
        return 0.0

# -----------------------------------------------------------------------------
# 4. RUN EVALUATION
# -----------------------------------------------------------------------------
results = []
os.makedirs("eval_audio", exist_ok=True)

print("\n--- Starting Generation ---")

for idx, text in enumerate(test_sentences):
    print(f"▶️ Processing {idx+1}/{len(test_sentences)}...")
    
    out_wav_path = f"eval_audio/sample_{idx}.wav"
    
    # Generate
    outputs = model.synthesize(text, config, speaker_wav=REF_AUDIO_PATH, gpt_cond_len=3, language="en")
    torchaudio.save(out_wav_path, torch.tensor(outputs["wav"]).unsqueeze(0), 24000)

    # WER
    transcription = asr_model.transcribe(out_wav_path, language="en")["text"]
    error_rate = wer(text.lower(), transcription.lower().strip())

    # Similarity
    sim_score = get_similarity(out_wav_path, REF_AUDIO_PATH)

    print(f"   Saved: {out_wav_path} | WER: {error_rate:.2f} | Sim: {sim_score:.2f}")

    results.append({
        "Original": text,
        "Transcribed": transcription,
        "WER": round(error_rate, 4),
        "Similarity": round(sim_score, 4)
    })

# -----------------------------------------------------------------------------
# 5. SAVE
# -----------------------------------------------------------------------------
if results:
    df = pd.DataFrame(results)
    print("\n" + "="*30)
    print(f"📊 FINAL SCORE")
    print(f"Avg WER: {df['WER'].mean():.4f} (Lower is better)")
    print(f"Avg Similarity: {df['Similarity'].mean():.4f} (Higher is better)")
    print("="*30)
    df.to_csv(RESULTS_FILE, index=False)