import os
import shutil
import subprocess
import uuid
import threading

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from app.pipeline.dubbing_pipeline import process_video

router = APIRouter(
    prefix="/video",
    tags=["Video"]
)

UPLOAD_FOLDER = "uploads/videos"
PROCESSED_FOLDER = "uploads/processed"
OUTPUT_FOLDER = "outputs/final"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

FFMPEG_PATH = r"D:\ffmpeg\bin\ffmpeg.exe"

# =====================================================
# Store Jobs
# =====================================================

jobs = {}


# =====================================================
# Update Job Status
# =====================================================

def update_job(
    job_id: str,
    step: str,
    progress: int,
    status: str = "processing"
):
    if job_id not in jobs:
        return

    jobs[job_id]["status"] = status
    jobs[job_id]["step"] = step
    jobs[job_id]["progress"] = progress

    print("=" * 60)
    print("JOB      :", job_id)
    print("STATUS   :", status)
    print("STEP     :", step)
    print("PROGRESS :", f"{progress}%")
    print("=" * 60)

# =====================================================
# Preprocess Video
# =====================================================

def preprocess_video(input_path, output_path):
    subprocess.run(
        [
            FFMPEG_PATH,
            "-y",
            "-i",
            input_path,
            "-vf",
            "scale='min(1920,iw)':-2",
            "-r",
            "25",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            output_path
        ],
        check=True
    )

# =====================================================
# Background Pipeline
# =====================================================

def run_pipeline(
    job_id,
    processed_path,
    language
):
    try:
        update_job(
            job_id,
            "Starting AI Pipeline",
            15
        )

        # FIXED: Explicitly passing job_id to pipeline
        output_path = process_video(
            video_path=processed_path,
            target_language=language,
            job_id=job_id,
            progress_callback=lambda step, progress:
                update_job(
                    job_id,
                    step,
                    progress
                )
        )

        if not os.path.exists(output_path):
            raise Exception(
                "Final video not generated."
            )

        output_path = output_path.replace(
            "\\",
            "/"
        )

        jobs[job_id] = {
            "status": "completed",
            "step": "Completed",
            "progress": 100,
            "video_url": f"http://127.0.0.1:8000/{output_path}",
            "error": None
        }

        print("=" * 60)
        print("PIPELINE FINISHED")
        print(jobs[job_id]["video_url"])
        print("=" * 60)

    except Exception as e:
        print("=" * 60)
        print("PIPELINE ERROR")
        print(str(e))
        print("=" * 60)

        jobs[job_id] = {
            "status": "failed",
            "step": "Failed",
            "progress": 100,
            "video_url": None,
            "error": str(e)
        }

# =====================================================
# Upload Endpoint
# =====================================================

@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    language: str = Form(...)
):
    try:
        job_id = str(uuid.uuid4())

        jobs[job_id] = {
            "status": "processing",
            "step": "Uploading Video",
            "progress": 0,
            "video_url": None,
            "error": None
        }
        input_path = os.path.join(
            UPLOAD_FOLDER,
            file.filename
        )
        with open(
            input_path,
            "wb"
        ) as buffer:
            shutil.copyfileobj(
                file.file,
                buffer
            )
        update_job(
            job_id,
            "Preprocessing Video",
            10
        )
        processed_path = os.path.join(
            PROCESSED_FOLDER,
            file.filename
        )
        preprocess_video(
            input_path,
            processed_path
        )
        thread = threading.Thread(
            target=run_pipeline,
            args=(
                job_id,
                processed_path,
                language
            ),
            daemon=True
        )
        thread.start()
        return JSONResponse(
            {
                "success": True,
                "job_id": job_id,
                "status": "processing"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# =====================================================
# Status Endpoint
# =====================================================

@router.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in jobs:
        return {
            "status": "not_found",
            "job_id": job_id
        }
    return jobs[job_id]