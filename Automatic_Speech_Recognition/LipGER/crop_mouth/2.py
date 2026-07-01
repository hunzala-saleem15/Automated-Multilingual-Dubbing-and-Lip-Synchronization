import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import face_alignment

# =====================================================
# CONFIG
# =====================================================
FEMALE_VIDEO_ROOT = r"E:\ASR\facestar\female_speaker"
OUTPUT_ROOT = r"E:\ASR\facestar\MouthCrops_Female"

INPUT_TRAIN = "trainset"
INPUT_TEST  = "testset"

OUTPUT_TRAIN_DIR = os.path.join(OUTPUT_ROOT, "train")
OUTPUT_TEST_DIR  = os.path.join(OUTPUT_ROOT, "test")
FAILED_FRAMES_DIR = os.path.join(OUTPUT_ROOT, "failed_frames")

for d in [OUTPUT_TRAIN_DIR, OUTPUT_TEST_DIR, FAILED_FRAMES_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# Portrait output size 9:16
# =====================================================
OUTPUT_WIDTH = 256
OUTPUT_HEIGHT = 456
OUTPUT_SIZE = (OUTPUT_WIDTH, OUTPUT_HEIGHT)

PAD_MOUTH = 6
NUM_FRAMES = 50

# =====================================================
# Initialize FAN
# =====================================================
fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    device=DEVICE,
    flip_input=False
)

# =====================================================
# Helper: select NUM_FRAMES uniformly
# =====================================================
def select_frames(total_frames, num_frames=NUM_FRAMES):
    if total_frames <= num_frames:
        return list(range(total_frames))
    return np.linspace(0, total_frames - 1, num_frames, dtype=int)

# =====================================================
# Helper: crop mouth only + 9:16 ratio
# =====================================================
def crop_mouth_centered(frame, landmarks, output_size=OUTPUT_SIZE, pad=PAD_MOUTH):
    mouth_points = landmarks[48:68]
    x, y, w, h = cv2.boundingRect(np.array(mouth_points))

    x_min, y_min = max(0, x - pad), max(0, y - pad)
    x_max, y_max = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2

    target_w = x_max - x_min
    target_h = int(target_w * 16 / 9)

    y_min_new = max(0, cy - target_h // 2)
    y_max_new = min(frame.shape[0], cy + target_h // 2)

    crop = frame[y_min_new:y_max_new, x_min:x_max]
    if crop.size == 0:
        return None
    return cv2.resize(crop, output_size)

# =====================================================
# Collect all mp4 videos
# =====================================================
def collect_videos(root_dir):
    return [
        os.path.join(r, f)
        for r, _, files in os.walk(root_dir)
        for f in files
        if f.lower().endswith(".mp4") and "_crop" not in f
    ]

# =====================================================
# Process single video
# =====================================================
def process_video(video_path, output_dir):
    name = os.path.basename(video_path)
    out_path = os.path.join(output_dir, f"{os.path.splitext(name)[0]}_crop.mp4")

    if os.path.exists(out_path):
        print(f"[SKIP] {name}")
        return True

    print(f"\n▶ Processing video: {name}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_ids = select_frames(total_frames)

    fps = NUM_FRAMES / 3.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(out_path, fourcc, fps, OUTPUT_SIZE)

    prev_frame = None
    for idx in tqdm(frame_ids, desc=f"🎥 {name}", ncols=90):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        try:
            landmarks_all = fa.get_landmarks(frame)
            if landmarks_all:
                crop = crop_mouth_centered(frame, landmarks_all[0])
                if crop is not None:
                    prev_frame = crop

            if prev_frame is not None:
                out_vid.write(prev_frame)
            else:
                fail_img = os.path.join(FAILED_FRAMES_DIR, f"{name}_frame{idx}.jpg")
                cv2.imwrite(fail_img, frame)

        except Exception as e:
            print(f"\n[WARN] {name} frame {idx}: {e}")

    cap.release()
    out_vid.release()
    print(f"✅ Done: {name} ({len(frame_ids)} frames saved)")
    return True

# =====================================================
# MAIN LOOP: process train + test
# =====================================================
all_videos = []
processed_videos = []

for split, out_dir in [(INPUT_TRAIN, OUTPUT_TRAIN_DIR), (INPUT_TEST, OUTPUT_TEST_DIR)]:
    split_path = os.path.join(FEMALE_VIDEO_ROOT, split)
    if not os.path.exists(split_path):
        print(f"[SKIP] {split_path} does not exist")
        continue

    videos = collect_videos(split_path)
    print(f"\n🚀 Processing {split.upper()} — {len(videos)} videos")
    all_videos.extend(videos)

    for v in videos:
        success = process_video(v, out_dir)
        if success:
            processed_videos.append(v)

# =====================================================
# VERIFY: check all videos processed
# =====================================================
print("\n🔎 Verification:")
print(f"Total videos found : {len(all_videos)}")
print(f"Total videos cropped: {len(processed_videos)}")
missing = set(all_videos) - set(processed_videos)
if missing:
    print(f"❌ Videos failed/skipped: {len(missing)}")
    for m in missing:
        print("   ", m)
else:
    print("✅ All videos processed successfully!")

print("\n🎉 FEMALE VIDEOS CROPPED SUCCESSFULLY (Mouth only, 9:16, ~3 sec, 50 frames)")
