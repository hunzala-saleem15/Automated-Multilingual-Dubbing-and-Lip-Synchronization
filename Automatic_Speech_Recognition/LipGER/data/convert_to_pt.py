import json
import random
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# -----------------------------
# Reproducibility
# -----------------------------
random.seed(42)
torch.manual_seed(42)

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[+] Using device:", device)

# -----------------------------
# Paths
# -----------------------------
json_files = [
    r"E:\ASR\facestar_whisper\facestar_full_train_whisper_fixed_clean_backup.json",
    r"E:\ASR\facestar_whisper\facestar_full_test_whisper_fixed_clean_backup.json"
]

pt_files = [
    r"E:\ASR\facestar_whisper\facestar_full_train_whisper_fixed_clean.pt",
    r"E:\ASR\facestar_whisper\facestar_full_test_whisper_fixed_clean.pt"
]

# -----------------------------
# Models
# -----------------------------
model_name = "microsoft/phi-2"
embedding_model_name = "all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(embedding_model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side="left")

# -----------------------------
# Helper functions
# -----------------------------
def random_sample_sequence(lst, sample_size):
    return random.sample(lst, min(sample_size, len(lst)))

def calculate_noise_embedding(n_best_hypotheses):
    if len(n_best_hypotheses) == 0:
        return torch.zeros((1, embedding_model.get_sentence_embedding_dimension()))
    if len(n_best_hypotheses) < 2:
        emb = embedding_model.encode(n_best_hypotheses, convert_to_tensor=True)
        return torch.zeros((1, emb.shape[1]))
    embeddings = embedding_model.encode(n_best_hypotheses, convert_to_tensor=True)
    diffs = embeddings[:, None, :] - embeddings[None, :, :]
    idx = torch.tril_indices(diffs.shape[0], diffs.shape[1], offset=-1)
    return diffs[idx[0], idx[1]].cpu()

# -----------------------------
# Prompts
# -----------------------------
prompt_1 = (
    "Below is the best-hypotheses transcribed from speech recognition system. "
    "Please try to revise it using the words which are only included into other-hypothesis, "
    "and write the response for the true transcription.\n\n### Best-hypothesis:\n"
)
prompt_2 = "\n\n### Other-hypothesis:"
prompt_3 = "\n\n### Response:\n"

# -----------------------------
# Main Conversion Loop
# -----------------------------
for json_file, pt_file in zip(json_files, pt_files):
    print("[+] Processing:", json_file)

    with open(json_file, "r", encoding="utf-8") as f:
        all_files = json.load(f)

    all_pt = []

    for i, item in enumerate(tqdm(all_files, desc="Converting JSON → PT")):
        new_dict = {"id": f"sample_{i}"}

        # Ground truth / Best hypothesis
        best_hyp = item.get("Caption", "").strip().lower()
        if best_hyp == "":
            print(f"[!] Skipping sample {i}: Empty Caption")
            continue

        # Other hypotheses (nhyps_base)
        n_best_hyps = [h.lower() for h in item.get("nhyps_base", []) if isinstance(h, str)]
        sampled_hyps = random_sample_sequence(n_best_hyps, 5)

        # Construct prompts
        final_prompt_no_response = prompt_1 + best_hyp + prompt_2 + "\n" + "\n".join(sampled_hyps) + prompt_3
        final_prompt = final_prompt_no_response + best_hyp + "<|endoftext|>"

        # Tokenization
        input_ids_no_response = tokenizer(final_prompt_no_response, return_tensors="pt").input_ids.squeeze(0)
        input_ids = tokenizer(final_prompt, return_tensors="pt").input_ids.squeeze(0)

        # Labels only for response part
        labels = torch.full_like(input_ids, -100)
        labels[len(input_ids_no_response):] = input_ids[len(input_ids_no_response):]

        new_dict["input_ids_no_response"] = input_ids_no_response
        new_dict["input_ids"] = input_ids
        new_dict["labels"] = labels
        new_dict["input"] = final_prompt
        new_dict["ground_truth"] = best_hyp

        # Noise embeddings
        new_dict["emb_diff"] = calculate_noise_embedding(n_best_hyps)

        # Speech features (whisper embedding)
        whisper_emb_path = item.get("whisper_emb", None)
        if whisper_emb_path is None:
            print(f"[!] Skipping sample {i}: No whisper_emb path")
            continue
        if not os.path.isfile(whisper_emb_path):
            print(f"[!] Skipping sample {i}: File not found: {whisper_emb_path}")
            continue
        try:
            audio_features = np.load(whisper_emb_path)
            audio_features = torch.tensor(audio_features, dtype=torch.float)
            new_dict["clean_speech"] = audio_features
            new_dict["noisy_speech"] = audio_features.clone()
        except Exception as e:
            print(f"[!] Skipping sample {i}: Error loading {whisper_emb_path} | {e}")
            continue

        # Mouth ROI (optional)
        mouthroi_path = item.get("Mouthroi", None)
        if mouthroi_path is not None and os.path.isfile(mouthroi_path):
            new_dict["mouthroi"] = mouthroi_path
        else:
            new_dict["mouthroi"] = None

        all_pt.append(new_dict)

    # Save PT file
    torch.save(all_pt, pt_file)
    print(f"[+] Saved {pt_file} | Samples: {len(all_pt)}")