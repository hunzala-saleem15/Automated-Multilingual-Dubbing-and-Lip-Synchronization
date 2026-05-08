import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# -----------------------------------------------------------------------------
# 1. PATHS (Winner File Updated)
# -----------------------------------------------------------------------------
TRAIN_DIR = r"E:\TTS\TTS\recipes\ljspeech\xtts_v2\run\training\GPT_XTTS_v2_English_FineTune-January-17-2026_10+42PM-dbf1a08a"
CONFIG_PATH = os.path.join(TRAIN_DIR, "config.json")
MODEL_PATH = os.path.join(TRAIN_DIR, "best_model.pth") 
VOCAB_PATH = r"E:\TTS\TTS\checkpoints\XTTS_v2\vocab.json"

# 🏆 THE WINNER REFERENCE AUDIO
WINNER_REF = r"E:\TTS\TTS\data\wavs\1246_135815_000038_000003.wav"

# Final Showcase Text (Thoda mushkil aur lamba)
TEXT = """
This is the final demonstration of my custom trained voice model. 
By selecting the correct reference audio, we achieved a similarity score of over zero point seven one, which is an excellent result.
The voice should now sound stable, clear, and very close to the original speaker's identity. 
I am now ready to use this model for content creation, audiobooks, or assistants.
"""

# -----------------------------------------------------------------------------
# 2. GENERATION
# -----------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Generating Final Audio on {device}...")

config = XttsConfig()
config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=MODEL_PATH, vocab_path=VOCAB_PATH, use_deepspeed=False)
model.to(device)

print("🎙️ Synthesizing...")

# Generate
out = model.synthesize(
    TEXT,
    config,
    speaker_wav=WINNER_REF,
    gpt_cond_len=3,
    language="en",
    temperature=0.7, # Creativity control
    top_p=0.85,      # Stability control
    top_k=50,        # Stability control
)

# Save
output_path = "Final_Showcase_Voice.wav"
torchaudio.save(output_path, torch.tensor(out["wav"]).unsqueeze(0), 24000)

print("-" * 40)
print(f"🎉 COMPLETED! Audio saved as: {output_path}")
print("-" * 40)