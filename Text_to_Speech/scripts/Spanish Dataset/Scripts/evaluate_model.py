import os
import torch
import torchaudio
import whisper
import pandas as pd
from jiwer import wer
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch.nn.functional as F

# SpeechBrain Import (Compatible with v0.5.16)
try:
    from speechbrain.pretrained import EncoderClassifier
except ImportError:
    from speechbrain.inference.speaker import EncoderClassifier

# -----------------------------------------------------------------------------
# 1. PATHS SETUP
# -----------------------------------------------------------------------------
TRAIN_DIR = r"E:\TTS\TTS\recipes\ljspeech\xtts_v2\run\training\GPT_XTTS_v2_Spanish_FineTune-January-16-2026_07+43PM-dbf1a08a"
CONFIG_PATH = os.path.join(TRAIN_DIR, "config.json")
MODEL_PATH = os.path.join(TRAIN_DIR, "best_model.pth")
VOCAB_PATH = r"E:\TTS\TTS\checkpoints\XTTS_v2\vocab.json"
REF_AUDIO_PATH = r"E:\TTS\TTS\data\wavs\angelina\angelina_00_delgado_f000001.wav"

RESULTS_FILE = "evaluation_results.csv"

test_sentences = [
    "Hola, esta es una prueba para calcular la tasa de error de palabras.",
    "La inteligencia artificial está cambiando el mundo rápidamente.",
    "Espero que este experimento funcione correctamente para mi investigación.",
    "La clonación de voz permite preservar la identidad del hablante.",
    "El clima hoy está muy agradable para salir a caminar."
]

# -----------------------------------------------------------------------------
# 2. LOAD MODELS
# -----------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running Generation on: {device}")

# A. Load XTTS (Generation on GPU for Speed)
print("Loading XTTS Model...")
config = XttsConfig()
config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=MODEL_PATH, vocab_path=VOCAB_PATH, use_deepspeed=False)
model.to(device)

# B. Load Whisper (ASR on GPU)
print("Loading Whisper (ASR)...")
asr_model = whisper.load_model("base", device=device)

# C. Load SpeechBrain (Similarity on CPU to avoid Tensor Errors)
print("Loading Speaker Encoder (Similarity on CPU)...")
# We force CPU here to ensure stability
spk_encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", 
    run_opts={"device": "cpu"} 
)

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def get_similarity(audio_path1, audio_path2):
    """Calculates Cosine Similarity manually to avoid Shape Errors"""
    try:
        # Load Audio
        signal1, fs1 = torchaudio.load(audio_path1)
        signal2, fs2 = torchaudio.load(audio_path2)
        
        # Convert to Mono
        if signal1.shape[0] > 1: signal1 = torch.mean(signal1, dim=0, keepdim=True)
        if signal2.shape[0] > 1: signal2 = torch.mean(signal2, dim=0, keepdim=True)

        # Resample to 16k
        if fs1 != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=fs1, new_freq=16000)
            signal1 = resampler(signal1)
        if fs2 != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=fs2, new_freq=16000)
            signal2 = resampler(signal2)

        # Ensure CPU
        signal1 = signal1.cpu()
        signal2 = signal2.cpu()

        # Get Embeddings
        with torch.no_grad():
            emb1 = spk_encoder.encode_batch(signal1)
            emb2 = spk_encoder.encode_batch(signal2)

        # --- FORCE FLATTEN & MANUAL COSINE SIMILARITY ---
        # Ye tarika dimensions ke masle ko 100% khatam kar deta hai
        vec1 = emb1.reshape(-1)
        vec2 = emb2.reshape(-1)

        # Formula: (A . B) / (|A| * |B|)
        dot_product = torch.dot(vec1, vec2)
        norm1 = torch.norm(vec1)
        norm2 = torch.norm(vec2)

        similarity = dot_product / (norm1 * norm2)
        
        return similarity.item()

    except Exception as e:
        print(f"Warning: Similarity check failed. Error: {e}")
        return 0.0

# -----------------------------------------------------------------------------
# 4. RUN EVALUATION
# -----------------------------------------------------------------------------
results = []
os.makedirs("eval_audio", exist_ok=True)

print("\n--- Starting Evaluation ---")

for idx, text in enumerate(test_sentences):
    print(f"Processing {idx+1}/{len(test_sentences)}: {text[:30]}...")
    
    # 1. Generate Audio
    out_wav_path = f"eval_audio/sample_{idx}.wav"
    outputs = model.synthesize(text, config, speaker_wav=REF_AUDIO_PATH, gpt_cond_len=3, language="es")
    
    # Save audio
    torchaudio.save(out_wav_path, torch.tensor(outputs["wav"]).unsqueeze(0), 24000)

    # 2. Calculate WER
    transcription_result = asr_model.transcribe(out_wav_path, language="es")
    transcribed_text = transcription_result["text"]
    error_rate = wer(text.lower(), transcribed_text.lower())

    # 3. Calculate Similarity
    sim_score = get_similarity(out_wav_path, REF_AUDIO_PATH)

    print(f"   -> WER: {error_rate:.4f} | Similarity: {sim_score:.4f}")

    results.append({
        "Original Text": text,
        "Transcribed Text": transcribed_text,
        "WER": round(error_rate, 4),
        "Similarity Score": round(sim_score, 4)
    })

# -----------------------------------------------------------------------------
# 5. SAVE RESULTS
# -----------------------------------------------------------------------------
if results:
    df = pd.DataFrame(results)
    avg_wer = df["WER"].mean()
    avg_sim = df["Similarity Score"].mean()

    print("\n" + "="*40)
    print(f"FINAL RESULTS")
    print("="*40)
    print(f"Average WER (Lower is better): {avg_wer:.4f}")
    print(f"Average Similarity (Higher is better): {avg_sim:.4f}")
    print("="*40)

    df.to_csv(RESULTS_FILE, index=False)
    print(f"Detailed results saved to {RESULTS_FILE}")
else:
    print("No results generated.")