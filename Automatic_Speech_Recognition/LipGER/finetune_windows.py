# finetune_windows.py (Windows safe)
import subprocess
import sys
import os

# -----------------------------
# 🔹 Paths
# -----------------------------
venv_path = r'E:\ASR\.venv'
python_exe = os.path.join(venv_path, 'Scripts', 'python.exe')

train_path = r'E:\ASR\facestar_whisper\formatted_json\data\mciro_train_whisper_tiny.pt'
val_path  = r'E:\ASR\facestar_whisper\formatted_json\data\mciro_test_whisper_tiny.pt'
checkpoint_dir = r'E:\ASR\TinyLlama\TinyLlama-1.1B-3T'

# -----------------------------
# 🔹 Hyperparameters
# -----------------------------
device = 'cuda'
batch_size = 16
epochs = 2
learning_rate = 0.0001

# -----------------------------
# 🔹 Step 1: Create virtual environment if not exists
# -----------------------------
if not os.path.exists(venv_path):
    print(f"Creating virtual environment at {venv_path}...")
    subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)

# -----------------------------
# 🔹 Step 2: Install required packages inside venv
# -----------------------------
packages = [
    "numpy<2",  # avoid numpy 2.x issues
    "torch>=2.1.0.dev0",
    "lightning>=2.1.0.dev0",
    "pytorch_lightning>=2.1.0.dev0",
    "transformers",
    "h5py",
    "tqdm",
    "typing_extensions",
    "xformers"  # optional
]

print("Upgrading pip and installing packages inside venv...")
subprocess.run([python_exe, "-m", "pip", "install", "--upgrade", "pip"], check=True)
subprocess.run([python_exe, "-m", "pip", "install"] + packages, check=True)

# -----------------------------
# 🔹 Step 3: Run LipGER fine-tuning
# -----------------------------
lipger_script = r'E:\ASR\LipGER\finetune\lipger.py'
cmd = [
    python_exe, lipger_script,
    "--data", "facestar_whisper",
    "--train_path", train_path,
    "--val_path", val_path,
    "--checkpoint_dir", checkpoint_dir,
    "--device", device,
    "--batch_size", str(batch_size),
    "--epochs", str(epochs),
    "--lr", str(learning_rate)
]

# Set PYTHONPATH to LipGER root to avoid ModuleNotFoundError
env = os.environ.copy()
env["PYTHONPATH"] = r"E:\ASR\LipGER"

print("Running LipGER fine-tuning...")
subprocess.run(cmd, check=True, env=env)
