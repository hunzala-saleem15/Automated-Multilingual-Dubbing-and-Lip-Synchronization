import os
import shutil
import subprocess

from app.models_ai.preprocessing import preprocessor
from app.models_ai.asr import asr_model
from app.models_ai.nllb import translator
from app.models_ai.xtts import tts_model
from app.models_ai.talklip import talklip_model

TEMP_DIR = "outputs/temp"
FINAL_DIR = "outputs/final"

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

LANG_MAP = {
    "Arabic": (
        "eng_Latn",
        "arb_Arab",
        "ar"
    ),
    "Spanish": (
        "eng_Latn",
        "spa_Latn",
        "es"
    )
}


def process_video(
    video_path,
    target_language,
    job_id=None,
    progress_callback=None
):
    from app.routes.video_routes import update_job
    print("=" * 60)
    print("PIPELINE STARTED")
    print("=" * 60)

    base = os.path.splitext(
        os.path.basename(video_path)
    )[0]

    audio_path = os.path.join(
        TEMP_DIR,
        base + ".wav"
    )

    tts_audio = os.path.join(
        TEMP_DIR,
        base + "_tts.wav"
    )

    final_video = os.path.join(
        FINAL_DIR,
        base + "_dubbed.mp4"
    )

    ##################################################
    # STEP 1: Extract Audio
    ##################################################

    if progress_callback:
        progress_callback(
            "Extracting Audio",
            20
        )

    print("\nSTEP 1 : Extract Audio")

    preprocessor.extract_audio(
        video_path,
        audio_path
    )

    ##################################################
    # STEP 2: Whisper ASR
    ##################################################

    if progress_callback:
        progress_callback(
            "Running Whisper ASR",
            35
        )

    print("\nSTEP 2 : Whisper ASR")

    asr = asr_model.transcribe(
        audio_path
    )

    transcript = asr["text"]

    ##################################################
    # STEP 3: Translation
    ##################################################

    if progress_callback:
        progress_callback(
            "Translating Text",
            50
        )

    print("\nSTEP 3 : Translation")

    src, tgt, xtts_lang = LANG_MAP[target_language]

    translated = translator.translate(
        transcript,
        source_lang=src,
        target_lang=tgt
    )

    ##################################################
    # STEP 4: XTTS
    ##################################################

    if progress_callback:
        progress_callback(
            "Generating AI Voice",
            65
        )

    print("\nSTEP 4 : XTTS")

    tts_model.generate_speech(
        text=translated,
        speaker_wav=audio_path,
        language=xtts_lang,
        output_path=tts_audio
    )

    ##################################################
    # STEP 5: BBX Extraction
    ##################################################

    if progress_callback:
        progress_callback(
            "Preparing Face Detection",
            75
        )

    print("\nSTEP 5 : BBX Extraction")

    runtime = r"D:\Website\server\outputs\runtime"

    video_root = os.path.join(
        runtime,
        "videos"
    )

    bbx_root = os.path.join(
        runtime,
        "bbx"
    )

    os.makedirs(video_root, exist_ok=True)
    os.makedirs(bbx_root, exist_ok=True)

    sample = base

    runtime_video = os.path.join(
        video_root,
        sample + ".mp4"
    )

    shutil.copy(
        video_path,
        runtime_video
    )

    filelist = os.path.join(
        runtime,
        "filelist.txt"
    )

    with open(filelist, "w") as f:
        f.write(sample + "\n")

    # Debug checks to pinpoint WinError 2
    conda_bat_path = r"C:\Users\user\miniconda3\condabin\conda.bat"
    talklip_dir = r"D:\Website\server\Talklip"
    bbx_script_path = r"D:\Website\server\Talklip\preparation\bbx_extract.py"

    print("=" * 60)
    print("CONDABAT EXISTS :", os.path.exists(conda_bat_path))
    print("TALKLIP EXISTS  :", os.path.exists(talklip_dir))
    print("BBX EXISTS      :", os.path.exists(bbx_script_path))
    print("FILELIST EXISTS :", os.path.exists(filelist))
    print("VIDEO EXISTS    :", os.path.exists(runtime_video))
    print("=" * 60)

    result = subprocess.run(
        [
            conda_bat_path,
            "run",
            "-n",
            "talklip",
            "python",
            "preparation/bbx_extract.py",
            "--filelist",
            filelist,
            "--video_root",
            video_root,
            "--bbx_root",
            bbx_root,
            "--rank",
            "1",
            "--gpu",
            "0"
        ],
        cwd=talklip_dir,
        capture_output=True,
        text=True,
        check=False
    )

    print("=" * 60)
    print("STDOUT")
    print(result.stdout)
    print("=" * 60)
    print("STDERR")
    print(result.stderr)
    print("=" * 60)
    print("RETURN CODE:", result.returncode)

    if result.returncode != 0:
        raise Exception(f"BBX Extraction Failed with code {result.returncode}.\nSTDERR: {result.stderr}")

    bbx_folder = bbx_root
    bbx_file = os.path.join(bbx_root, "input.npy")

    if not os.path.isfile(bbx_file):
        raise Exception(f"BBX file not found: {bbx_file}")

    ##################################################
    # STEP 6: TalkLip
    ##################################################

    if progress_callback:
        progress_callback(
            "Lip Synchronization",
            90
        )

    print("\nSTEP 6 : TalkLip")

    talklip_model.generate_video(
        video_path,
        tts_audio,
        bbx_file,
        final_video,
        progress_callback=lambda step, progress: update_job(
            job_id,
            step,
            progress
        ) if job_id else None
    )

    ##################################################
    # FINAL
    ##################################################

    if progress_callback:
        progress_callback(
            "Rendering Final Video",
            98
        )

    print("=" * 60)
    print("PIPELINE FINISHED")
    print(final_video)
    print("=" * 60)

    if progress_callback:
        progress_callback(
            "Completed",
            100
        )

    return final_video