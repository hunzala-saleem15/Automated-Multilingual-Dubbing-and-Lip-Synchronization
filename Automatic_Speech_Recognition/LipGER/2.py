import os
import glob
import torch
import h5py
import torch.nn.functional as F
from lipger import Lipreading, device

# -----------------------------
# HDF5 folder containing all videos
# -----------------------------
hdf5_folder = r"E:\ASR\facestar\MouthCrops_Female\train"
hdf5_files = glob.glob(os.path.join(hdf5_folder, "*.hdf5"))

if len(hdf5_files) == 0:
    raise FileNotFoundError(f"No HDF5 files found in folder: {hdf5_folder}")

print(f"[INFO] Found {len(hdf5_files)} HDF5 files")

# -----------------------------
# Model parameters
# -----------------------------
tcn_options = {
    "kernel_size": [3],
    "num_layers": 4,
    "dropout": 0.5,
    "width_mult": 1,
    "dwpw": False
}

lip_encoder = Lipreading(
    hidden_dim=256,
    backbone_type='resnet',
    num_classes=500,
    relu_type='prelu',
    tcn_options=tcn_options,
    extract_feats=True
).to(device)

lip_encoder.eval()
print("[INFO] Model loaded successfully.")

# -----------------------------
# Loop over all HDF5 files
# -----------------------------
for hdf5_filepath in hdf5_files:
    print(f"\n[INFO] Processing: {hdf5_filepath}")

    with h5py.File(hdf5_filepath, 'r') as f:
        keys = list(f.keys())
        if len(keys) == 0:
            print(f"[WARNING] No datasets in {hdf5_filepath}, skipping")
            continue

        dataset_key = keys[0]  # automatically pick first dataset
        frames = torch.tensor(f[dataset_key][:])
        print(f"[INFO] Original shape from HDF5: {frames.shape}")

    # -----------------------------
    # Shape handling
    # -----------------------------
    if len(frames.shape) == 3:
        frames = frames.unsqueeze(0).unsqueeze(0).float()  # (1, 1, T, H, W)
    elif len(frames.shape) == 4:
        frames = frames.permute(3, 0, 1, 2).unsqueeze(0).float()  # (1, C, T, H, W)
    else:
        print(f"[WARNING] Unexpected frame shape: {frames.shape}, skipping")
        continue

    # -----------------------------
    # Resize to 112x112
    # -----------------------------
    B, C, T, H, W = frames.shape
    frames = frames.view(B * T, C, H, W)
    frames = F.interpolate(frames, size=(112, 112), mode="bilinear", align_corners=False)
    frames = frames.view(B, C, T, 112, 112)
    print(f"[INFO] Shape after resize: {frames.shape}")

    # -----------------------------
    # Forward pass
    # -----------------------------
    lengths = frames.size(2)

    with torch.no_grad():
        features = lip_encoder(frames.to(device), lengths)

    print(f"[INFO] Features shape: {features.shape}")

    # -----------------------------
    # Save features
    # -----------------------------
    save_path = os.path.splitext(hdf5_filepath)[0] + "_features.pt"
    torch.save(features.cpu(), save_path)
    print(f"[INFO] Features saved to: {save_path}")

print("\n[SUCCESS] All HDF5 files processed successfully.")
