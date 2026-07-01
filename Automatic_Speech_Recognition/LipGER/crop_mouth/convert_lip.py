import cv2
import numpy as np
import h5py
import json
import os
from tqdm import tqdm

def video_to_hdf5(video_path, hdf5_path):
    """Convert MP4 video to HDF5 grayscale frames"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with h5py.File(hdf5_path, 'w') as hdf5_file:
        dset = hdf5_file.create_dataset('video_frames', (frame_count, height, width), dtype='uint8')
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dset[frame_index] = gray_frame
            frame_index += 1

    cap.release()

# ----------------------------
# Paths
# ----------------------------
json_files = [
    r"E:\ASR\facestar_whisper\formatted_json\mciro_train_whisper_tiny_formatted.json",
    r"E:\ASR\facestar_whisper\formatted_json\mciro_test_whisper_tiny_formatted.json"
]

female_videos_folder = r"E:\ASR\facestar\MouthCrops_Female"
male_videos_folder = r"E:\ASR\facestar\MouthCrops_Male"

# ----------------------------
# Helper function to get full path
# ----------------------------
def get_video_full_path(video_name, gender_folder):
    for subfolder in ['train', 'test']:
        candidate = os.path.join(gender_folder, subfolder, video_name)
        if os.path.exists(candidate):
            return candidate
    return None

# ----------------------------
# Process each JSON file
# ----------------------------
for json_file in json_files:
    print(f"\nProcessing JSON: {json_file}")
    with open(json_file, 'r') as f:
        dataset = json.load(f)

    for item in tqdm(dataset):
        video_name = os.path.basename(item['Mouthroi'])  # extract filename
        gender = item.get('Gender', 'Female')  # default Female if not specified
        if gender.lower() == 'female':
            video_path = get_video_full_path(video_name, female_videos_folder)
        else:
            video_path = get_video_full_path(video_name, male_videos_folder)

        if video_path is None:
            print(f"Video not found: {video_name}")
            continue

        hdf5_path = video_path.replace(".mp4", ".hdf5")
        item['Mouthroi'] = hdf5_path  # update JSON path

        if not os.path.exists(hdf5_path):
            print(f"Converting: {video_path} → {hdf5_path}")
            video_to_hdf5(video_path, hdf5_path)

    # Save updated JSON
    with open(json_file, 'w') as f:
        json.dump(dataset, f, indent=4)

print("✅ All videos converted to HDF5 and JSON files updated!")
