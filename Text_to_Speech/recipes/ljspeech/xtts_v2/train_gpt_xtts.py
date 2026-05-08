import os
import re   # <--- FIX 1: Text cleaning ke liye library add ki
import random
from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig

# -------------------------------------------------------------------------
# 1. PATHS SETUP
# -------------------------------------------------------------------------
RUN_NAME = "GPT_XTTS_v2_English_FineTune"
PROJECT_NAME = "XTTS_trainer"
DASHBOARD_LOGGER = "tensorboard"
LOGGER_URI = None

# Output Path
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run", "training")
os.makedirs(OUT_PATH, exist_ok=True)

# Data Paths
DATA_PATH = "E:/TTS/TTS/data"
MANIFEST_PATH = "E:/TTS/TTS/data/metadata.csv"

# Checkpoints Paths
DVAE_CHECKPOINT = "E:/TTS/TTS/checkpoints/XTTS_v2/dvae.pth"
MEL_NORM_FILE = "E:/TTS/TTS/checkpoints/XTTS_v2/mel_stats.pth"
XTTS_CHECKPOINT = "E:/TTS/TTS/checkpoints/XTTS_v2/model.pth"
TOKENIZER_FILE = "E:/TTS/TTS/checkpoints/XTTS_v2/vocab.json"

# -------------------------------------------------------------------------
# 2. CUSTOM FORMATTER (FIXED: Auto-Clean Text)
# -------------------------------------------------------------------------
def custom_formatter(root_path, manifest_file):
    """
    Reads metadata and CLEANS text (removes brackets like [his]).
    """
    items = []
    if not os.path.exists(manifest_file):
        print(f"ERROR: Metadata file not found at {manifest_file}")
        return []

    with open(manifest_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            
            if len(parts) < 2:
                continue
            
            wav_filename = parts[0].strip()
            raw_text = parts[1].strip()
            
            # --- FIX 1: CLEANING MAGIC ---
            # Remove brackets [] () and extra spaces
            clean_text = re.sub(r'[\[\]\(\)]', '', raw_text)
            clean_text = " ".join(clean_text.split())
            # -----------------------------

            speaker_name = "my_speaker"
            full_wav_path = os.path.join(root_path, "wavs", wav_filename + ".wav")

            items.append({
                "text": clean_text,
                "audio_file": full_wav_path,
                "speaker_name": speaker_name,
                "root_path": root_path,
                "language": "en"
            })
    return items

config_dataset = BaseDatasetConfig(
    formatter="ljspeech", 
    dataset_name="english_dataset",
    path=DATA_PATH,
    meta_file_train=MANIFEST_PATH,
    language="en",
)

DATASETS_CONFIG_LIST = [config_dataset]

# -------------------------------------------------------------------------
# 3. REFERENCE AUDIO FINDER
# -------------------------------------------------------------------------
ref_audio = "" 
if not os.path.exists(ref_audio):
    wavs_folder = os.path.join(DATA_PATH, "wavs")
    if os.path.exists(wavs_folder):
        for file in os.listdir(wavs_folder):
            if file.endswith(".wav"):
                ref_audio = os.path.join(wavs_folder, file)
                break

print(f"Reference Audio found: {ref_audio}")
SPEAKER_REFERENCE = [ref_audio] if ref_audio else []

# -------------------------------------------------------------------------
# 4. MAIN FUNCTION
# -------------------------------------------------------------------------
def main():
    print("Loading data manually...")
    all_samples = custom_formatter(DATA_PATH, MANIFEST_PATH)
    
    if len(all_samples) == 0:
        print("ERROR: No samples found!")
        return

    random.seed(42)
    random.shuffle(all_samples)
    
    eval_split_size = max(1, int(len(all_samples) * 0.01)) 
    eval_samples = all_samples[:eval_split_size]
    train_samples = all_samples[eval_split_size:]
    
    print(f" > Total Samples: {len(all_samples)}")
    print(f" > Train Samples: {len(train_samples)}")
    print(f" > Eval Samples:  {len(eval_samples)}")

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
        run_description="XTTS Fine-tuning English",
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        
        # --- FIX 2: STOPPING LOGIC ---
        epochs=10,           # Pehle 1000 tha. Ab ye sirf 5 loops chalega aur band ho jayega.
        # -----------------------------
        
        batch_size=2,
        batch_group_size=48,
        eval_batch_size=2,
        num_loader_workers=4,
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        
        log_model_step=500,  # Model jaldi save karega
        save_step=500,       # Har 500 steps pe checkpoint banayega
        
        save_n_checkpoints=1,
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

    model = GPTTrainer.init_from_config(config)

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
        train_samples=train_samples, 
        eval_samples=eval_samples,   
    )

    trainer.fit()

if __name__ == "__main__":
    main()