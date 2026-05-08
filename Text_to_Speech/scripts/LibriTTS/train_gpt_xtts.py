import os
import re
import random
from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig

# -------------------------------------------------------------------------
# 1. PATHS SETUP (UPDATED FOR LIBRITTS SPLITS)
# -------------------------------------------------------------------------
RUN_NAME = "LibriTTS_FineTune_v1" 
PROJECT_NAME = "XTTS_trainer"
DASHBOARD_LOGGER = "tensorboard"
LOGGER_URI = None

# Output Path
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run", "training")
os.makedirs(OUT_PATH, exist_ok=True)

# --- DATA PATHS ---
DATA_PATH = "E:/TTS/TTS/data"
# We now have specific files for Train and Eval
TRAIN_CSV = "E:/TTS/TTS/data/metadata_train.csv"
EVAL_CSV  = "E:/TTS/TTS/data/metadata_eval.csv"

# Checkpoints Paths (Base Model)
DVAE_CHECKPOINT = "E:/TTS/TTS/checkpoints/XTTS_v2/dvae.pth"
MEL_NORM_FILE = "E:/TTS/TTS/checkpoints/XTTS_v2/mel_stats.pth"
XTTS_CHECKPOINT = "E:/TTS/TTS/checkpoints/XTTS_v2/model.pth"
TOKENIZER_FILE = "E:/TTS/TTS/checkpoints/XTTS_v2/vocab.json"

# -------------------------------------------------------------------------
# 2. CUSTOM FORMATTER
# -------------------------------------------------------------------------
def custom_formatter(root_path, manifest_file):
    """
    Reads metadata and CLEANS text.
    """
    items = []
    if not os.path.exists(manifest_file):
        print(f"❌ ERROR: Metadata file not found at {manifest_file}")
        return []

    print(f"📄 Reading Metadata from: {manifest_file}")
    
    with open(manifest_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            
            if len(parts) < 2:
                continue
            
            wav_filename = parts[0].strip()
            raw_text = parts[1].strip()
            
            # Text Cleaning
            clean_text = re.sub(r'[\[\]\(\)]', '', raw_text)
            clean_text = " ".join(clean_text.split())

            speaker_name = "LibriTTS_Speaker" # Generic name for multispeaker
            
            # Ensure pointing to 'wavs' folder inside data
            full_wav_path = os.path.join(root_path, "wavs", wav_filename)

            items.append({
                "text": clean_text,
                "audio_file": full_wav_path,
                "speaker_name": speaker_name,
                "root_path": root_path,
                "language": "en"
            })
    return items

# Config for Training Set
config_dataset = BaseDatasetConfig(
    formatter=custom_formatter, 
    dataset_name="libritts_dataset",
    path=DATA_PATH,
    meta_file_train=TRAIN_CSV,
    language="en",
)

DATASETS_CONFIG_LIST = [config_dataset]

# -------------------------------------------------------------------------
# 3. REFERENCE AUDIO FINDER
# -------------------------------------------------------------------------
ref_audio = "" 
wavs_folder = os.path.join(DATA_PATH, "wavs")

# Try to pick a file specifically from the EVAL set for consistency during testing
if os.path.exists(EVAL_CSV) and os.path.exists(wavs_folder):
    try:
        with open(EVAL_CSV, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line:
                ref_filename = first_line.split('|')[0]
                ref_audio = os.path.join(wavs_folder, ref_filename)
    except:
        pass

# Fallback if specific file fails
if not ref_audio or not os.path.exists(ref_audio):
    if os.path.exists(wavs_folder):
        for file in os.listdir(wavs_folder):
            if file.endswith(".wav"):
                ref_audio = os.path.join(wavs_folder, file)
                break

print(f"🔍 Reference Audio for testing: {ref_audio}")
SPEAKER_REFERENCE = [ref_audio] if ref_audio else []

# -------------------------------------------------------------------------
# 4. MAIN FUNCTION
# -------------------------------------------------------------------------
def main():
    print("🚀 Initializing Trainer...")
    
    # Init Model Args
    model_args = GPTArgs(
        max_conditioning_length=132300,
        min_conditioning_length=66150,
        debug_loading_failures=False,
        max_wav_length=255995,
        max_text_length=200,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

    config = GPTTrainerConfig(
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="XTTS Fine-tuning on LibriTTS Clean",
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        
        # --- TRAINING SETTINGS ---
        epochs=10,            # 10 Epochs is good for fine-tuning
        batch_size=2,         # Low batch size to avoid GPU OOM
        batch_group_size=48,
        eval_batch_size=2,
        num_loader_workers=4,
        eval_split_max_size=256,
        print_step=10,        
        plot_step=100,
        
        log_model_step=100,   
        save_step=200,        
        
        save_n_checkpoints=2,
        save_checkpoints=True,
        print_eval=False,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[
            {
                "text": "It took me quite a long time to develop a voice, and now that I have it I am not going to be silent.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": "en"
            }
        ],
    )

    # Init Model
    model = GPTTrainer.init_from_config(config)

    # -----------------------------------------------------------------
    # LOAD DATASETS STRICTLY (NO RANDOM SPLITS)
    # -----------------------------------------------------------------
    print("\n📂 Loading Datasets...")
    
    # 1. Load Training Data
    print("   🔹 Loading Training Set...")
    train_samples = custom_formatter(DATA_PATH, TRAIN_CSV)
    
    # 2. Load Evaluation Data
    print("   🔹 Loading Evaluation Set...")
    eval_samples = custom_formatter(DATA_PATH, EVAL_CSV)
    
    if len(train_samples) == 0 or len(eval_samples) == 0:
        print("❌ ERROR: Samples missing! Check your CSV files.")
        return

    # No random shuffle/split needed here because the files are already split!
    
    print(f" > Total Train Samples: {len(train_samples)}")
    print(f" > Total Eval Samples:  {len(eval_samples)}")

    # Start Trainer
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=True,
            grad_accum_steps=16,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples, # Explicit Train List
        eval_samples=eval_samples,   # Explicit Eval List
    )

    print("🚀 Starting Training Loop...")
    trainer.fit()

if __name__ == "__main__":
    main()