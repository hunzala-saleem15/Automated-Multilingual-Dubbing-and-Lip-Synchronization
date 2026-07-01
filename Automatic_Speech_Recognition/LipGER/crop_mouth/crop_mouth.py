#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import av
import torch
import torchvision
from data_module import AVSRDataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings

# ==========================
# Suppress Torch JIT / warnings
# ==========================
warnings.filterwarnings("ignore")
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch.jit._state.disable()

# ==========================
# Paths & Settings
# ==========================
base_path = r"E:/ASR/facestar"
speakers = ["female_speaker", "male_speaker"]
splits = ["trainset", "testset"]

crop_folder = os.path.join(base_path, "cropped_videos")
os.makedirs(crop_folder, exist_ok=True)

max_workers = 2  # adjust according to your RAM/CPU

# ==========================
# Video crop function
# ==========================
def crop_video_file(data_filename, dst_filename, detector_type="retinaface", max_width=640):
    """
    Crop mouth region from a video safely (CPU safe)
    """
    # ------------------------------
    # Detector select
    # ------------------------------
    if detector_type == "mediapipe":
        from mediapipe.detector import LandmarksDetector
    else:
        from retinaface.detector import LandmarksDetector

    landmarks_detector = LandmarksDetector()  # CPU safe

    # ------------------------------
    # Load video frames frame-by-frame
    # ------------------------------
    frames = []
    container = av.open(data_filename)
    for frame in container.decode(video=0):
        img = frame.to_rgb().to_ndarray()
        h, w, _ = img.shape
        if w > max_width:
            scale = max_width / w
            new_h = int(h * scale)
            img = cv2.resize(img, (max_width, new_h))
        frames.append(img)

    # ------------------------------
    # Process landmarks per frame
    # ------------------------------
    landmarks_all = []
    for idx, f in enumerate(frames):
        try:
            landmarks_all.append(landmarks_detector(f))
        except Exception as e:
            print(f"[WARN] Frame {idx} failed: {e}")
            landmarks_all.append(None)
        if idx % 50 == 0:
            print(f"[INFO] Processed {idx}/{len(frames)} frames")

    # ------------------------------
    # Dataloader
    # ------------------------------
    dataloader = AVSRDataLoader(
        modality="video",
        speed_rate=1,
        transform=False,
        detector=detector_type,
        convert_gray=False
    )
    data = dataloader.load_data(data_filename, landmarks_all)

    # ------------------------------
    # Save cropped video
    # ------------------------------
    fps = cv2.VideoCapture(data_filename).get(cv2.CAP_PROP_FPS)
    os.makedirs(os.path.dirname(dst_filename), exist_ok=True)
    torchvision.io.write_video(dst_filename, data, fps)
    print(f"[DONE] Cropped video saved: {dst_filename}")


# ==========================
# Collect all mp4 files
# ==========================
mp4_files = []
for speaker in speakers:
    for split in splits:
        dir_path = os.path.join(base_path, speaker, split)
        if os.path.exists(dir_path):
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    if file.lower().endswith(".mp4"):
                        mp4_files.append(os.path.normpath(os.path.join(root, file)))

print(f"Total videos found: {len(mp4_files)}")

# ==========================
# Threaded cropping
# ==========================
errors = []

def crop_task(src_path):
    file_name = os.path.basename(src_path)
    dest_path = os.path.join(crop_folder, file_name[:-4] + "_crop.mp4")
    try:
        crop_video_file(src_path, dest_path, detector_type="retinaface")  # or "mediapipe"
        return (src_path, True, "")
    except Exception as e:
        return (src_path, False, str(e))


print("🔹 Cropping videos...")

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(crop_task, vf): vf for vf in mp4_files}
    for f in tqdm(as_completed(futures), total=len(futures), desc="Cropping videos"):
        src_path, success, err_msg = f.result()
        if not success:
            print(f"[ERROR] Failed: {src_path}\n{err_msg}")
            errors.append(src_path)

# ==========================
# Summary
# ==========================
print(f"\n✅ Cropping completed for {len(mp4_files) - len(errors)} videos.")
if errors:
    print(f"⚠️ Errors occurred for {len(errors)} videos. Check 'crop_errors.txt'")
    with open(os.path.join(base_path, "crop_errors.txt"), "w") as f:
        for e in errors:
            f.write(e + "\n")
