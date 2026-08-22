from dotenv import load_dotenv
import os

# Whisper
WHISPER_MODEL = "large-v3"

# Translation
MT_MODEL = "facebook/nllb-200-distilled-600M"

# XTTS
XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

# TalkLip
TALKLIP_ROOT = "D:/TalkLip"
TALKLIP_CHECKPOINT = "D:/TalkLip/checkpoints/checkpoint.pth"

# Folders
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

load_dotenv()

SAFEPAY_PUBLIC_KEY = os.getenv("SAFEPAY_PUBLIC_KEY")
SAFEPAY_SECRET_KEY = os.getenv("SAFEPAY_SECRET_KEY")