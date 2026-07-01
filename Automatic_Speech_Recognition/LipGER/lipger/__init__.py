# lipger/__init__.py

import os
import sys
import importlib.util
import torch

# -------------------------------
# Repo root (lipger folder) and sys.path setup
# -------------------------------
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.append(repo_root)

# -------------------------------
# Dynamically import lipreading_model.py
# -------------------------------
lip_model_path = os.path.join(repo_root, "lipreading_model.py")

if not os.path.exists(lip_model_path):
    raise FileNotFoundError(f"Lipreading model file not found: {lip_model_path}")

spec = importlib.util.spec_from_file_location("lipreading_model", lip_model_path)
lipreading_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lipreading_model)

# -------------------------------
# Expose Lipreading class publicly
# -------------------------------
Lipreading = lipreading_model.Lipreading

# -------------------------------
# Optional: default device
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Public API
# -------------------------------
__all__ = ["Lipreading", "device"]

# -------------------------------
# Info message
# -------------------------------
print(f"[INFO] LipGER initialized. Using device: {device}")
