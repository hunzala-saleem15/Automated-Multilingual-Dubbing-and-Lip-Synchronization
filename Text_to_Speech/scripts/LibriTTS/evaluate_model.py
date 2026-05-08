import os
import torch
import torchaudio
import whisper
import pandas as pd
import glob
import tqdm
from jiwer import wer
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# SpeechBrain Import
try:
    from speechbrain.pretrained import EncoderClassifier
except ImportError:
    from speechbrain.inference.speaker import EncoderClassifier

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

# 👇 Update this to your LATEST training folder
# 👇 PASTE YOUR ACTUAL FOLDER NAME HERE (No dots!)
TRAIN_DIR = r"E:\TTS\TTS\recipes\ljspeech\xtts_v2\run\training\LibriTTS_FineTune_v1-January-19-2026_03+14AM-dbf1a08a"

# Data Paths
DATA_ROOT = r"E:\TTS\TTS\data"
TEST_METADATA_PATH = r"E:\TTS\TTS\data\metadata_test.csv"
WAVS_DIR = os.path.join(DATA_ROOT, "wavs")

# Model Paths
CONFIG_PATH = os.path.join(TRAIN_DIR, "config.json")
VOCAB_PATH = r"E:\TTS\TTS\checkpoints\XTTS_v2\vocab.json"

# Output
RESULTS_FILE = "test_set_results.csv"
OUTPUT_AUDIO_DIR = "eval_output_audio"

# ⚠️ LIMIT SAMPLES (Set to None to run ALL 4000 lines, but it will take hours)
# For a quick test, keep it at 50 or 100.
NUM_SAMPLES = 50 

# ==============================================================================
# 2. AUTO-DETECT CHECKPOINT
# ==============================================================================
best_model = os.path.join(TRAIN_DIR, "best_model.pth")
if os.path.exists(best_model):
    MODEL_PATH = best_model
    print(f"✅ Found Best Model: {MODEL_PATH}")
else:
    checkpoints = glob.glob(os.path.join(TRAIN_DIR, "checkpoint_*.pth"))
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        MODEL_PATH = latest_checkpoint
        print(f"⚠️ 'best_model.pth' not found. Using latest checkpoint: {os.path.basename(MODEL_PATH)}")
    else:
        raise FileNotFoundError(f"❌ No models found in {TRAIN_DIR}. Train first!")

# ==============================================================================
# 3. LOAD MODELS
# ==============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on: {device}")

# A. XTTS
print("⏳ Loading XTTS...")
config = XttsConfig()
config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=MODEL_PATH, vocab_path=VOCAB_PATH, use_deepspeed=False)
model.to(device)

# B. Whisper (ASR)
print("⏳ Loading Whisper...")
asr_model = whisper.load_model("base", device=device)

# C. SpeechBrain (Similarity)
print("⏳ Loading Speaker Encoder...")
spk_encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", 
    run_opts={"device": "cpu"} # CPU is safer for this specific library to avoid conflicts
)

# ==============================================================================
# 4. HELPER FUNCTIONS
# ==============================================================================
def get_similarity(path1, path2):
    """Computes Cosine Similarity between two audio files"""
    try:
        # Load and prep
        sig1, fs1 = torchaudio.load(path1)
        sig2, fs2 = torchaudio.load(path2)
        
        # Mono & Resample to 16k (SpeechBrain requirement)
        if sig1.shape[0] > 1: sig1 = torch.mean(sig1, dim=0, keepdim=True)
        if sig2.shape[0] > 1: sig2 = torch.mean(sig2, dim=0, keepdim=True)
        
        if fs1 != 16000: sig1 = torchaudio.transforms.Resample(fs1, 16000)(sig1)
        if fs2 != 16000: sig2 = torchaudio.transforms.Resample(fs2, 16000)(sig2)

        # Encode
        emb1 = spk_encoder.encode_batch(sig1.cpu())
        emb2 = spk_encoder.encode_batch(sig2.cpu())

        # Similarity
        sim = torch.nn.functional.cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))
        return sim.item()
    except Exception as e:
        print(f"⚠️ Sim Error: {e}")
        return 0.0

# ==============================================================================
# 5. RUN EVALUATION LOOP
# ==============================================================================
print(f"\n📂 Reading Test Metadata: {TEST_METADATA_PATH}")
os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

results = []

with open(TEST_METADATA_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Filter lines (Remove header if exists, though usually metadata doesn't have one)
data_lines = [l.strip().split("|") for l in lines if "|" in l]

# Limit samples if requested
if NUM_SAMPLES:
    data_lines = data_lines[:NUM_SAMPLES]

print(f"📊 Processing {len(data_lines)} samples...")

# Progress Bar Loop
for idx, parts in enumerate(tqdm.tqdm(data_lines)):
    if len(parts) < 2: continue
    
    filename = parts[0].strip()
    text = parts[1].strip()
    
    # Paths
    original_audio_path = os.path.join(WAVS_DIR, filename)
    generated_audio_path = os.path.join(OUTPUT_AUDIO_DIR, f"gen_{filename}")
    
    if not os.path.exists(original_audio_path):
        print(f"⚠️ Missing file: {filename}")
        continue

    try:
        # 1. GENERATE
        # We use the ORIGINAL audio as the speaker reference
        outputs = model.synthesize(
            text, 
            config, 
            speaker_wav=original_audio_path, 
            gpt_cond_len=3, 
            language="en"
        )
        torchaudio.save(generated_audio_path, torch.tensor(outputs["wav"]).unsqueeze(0), 24000)

        # 2. CALCULATE WER
        transcription = asr_model.transcribe(generated_audio_path, language="en")["text"]
        wer_score = wer(text.lower(), transcription.lower().strip())

        # 3. CALCULATE SIMILARITY
        sim_score = get_similarity(generated_audio_path, original_audio_path)

        # Store Result
        results.append({
            "filename": filename,
            "text": text,
            "transcription": transcription,
            "WER": wer_score,
            "Similarity": sim_score
        })
        
    except Exception as e:
        print(f"❌ Error on {filename}: {e}")

# ==============================================================================
# 6. SAVE & SUMMARIZE
# ==============================================================================
if results:
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_FILE, index=False)
    
    print("\n" + "="*40)
    print("📢 FINAL TEST SET METRICS")
    print("="*40)
    print(f"Total Samples Evaluated: {len(df)}")
    print(f"Average WER (Lower is better):       {df['WER'].mean():.4f}")
    print(f"Average Similarity (Higher is better): {df['Similarity'].mean():.4f}")
    print("="*40)
    print(f"✅ Results saved to {RESULTS_FILE}")
else:
    print("❌ No results generated.")