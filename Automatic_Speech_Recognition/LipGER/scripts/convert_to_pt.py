import os
import json
import random
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

# -------------------------------
# Environment & Model Setup
# -------------------------------
os.environ['TRANSFORMERS_CACHE'] = '/fs/nexus-projects/brain_project/eccv/RobustGER/cache/hub'

# Sentence Transformer for noise embeddings
sbert_model = SentenceTransformer('all-MiniLM-L6-v2').to(torch.device("cuda"))

# HuggingFace LM model tokenizer
model_name = "microsoft/phi-2"  # Change if needed
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=True,
    access_token="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",
    padding_side='left'
)

# Maximum token length for the LM
MAX_LENGTH = 2048

# -------------------------------
# Helper Functions
# -------------------------------
def random_sample_sequence(lst, sample_size):
    if not lst:
        return []
    indices = random.sample(range(len(lst)), min(sample_size, len(lst)))
    indices.sort()
    return [lst[i] for i in indices]

def calculate_noise_embedding(n_best_hypotheses):
    if not n_best_hypotheses:
        return np.zeros((1, 384), dtype=np.float32)  # default embedding size for MiniLM
    embeddings = sbert_model.encode(n_best_hypotheses)
    embedding_diffs = []
    n = len(embeddings)
    for i in range(n):
        for j in range(i):
            embedding_diffs.append(embeddings[i] - embeddings[j])
    return np.stack(embedding_diffs, axis=0) if embedding_diffs else np.zeros((1, embeddings.shape[1]), dtype=np.float32)

# -------------------------------
# Prompts
# -------------------------------
prompt_1 = 'Below is the best-hypotheses transcribed from speech recognition system. Please try to revise it using the words which are only included into other-hypothesis, and write the response for the true transcription.\n\n### Best-hypothesis:\n'
prompt_2 = '\n\n### Other-hypothesis:'
prompt_3 = '\n\n### Response:\n'

# -------------------------------
# JSON Paths
# -------------------------------
train_json_path = r"E:\ASR\facestar_whisper\formatted_json\mciro_train_whisper_tiny_formatted.json"
test_json_path = r"E:\ASR\facestar_whisper\formatted_json\mciro_test_whisper_tiny_formatted.json"

output_folder = r"E:\ASR\facestar_whisper\formatted_json\data"
os.makedirs(output_folder, exist_ok=True)

# -------------------------------
# Conversion Function
# -------------------------------
def convert_json_to_pt(json_path, output_path):
    with open(json_path, 'r') as f:
        all_files = json.load(f)

    all_pt = []

    for item in tqdm(all_files, desc=f"Converting {os.path.basename(json_path)}"):
        new_dict = {'id': item['Uid']}

        # Lowercase all hypotheses and captions
        item["nhyps_base"] = [i.lower() for i in item["nhyps_base"]]
        item["Caption"] = item["Caption"].lower()

        # Randomly sample other hypotheses
        other_hypothesis = random_sample_sequence(item["nhyps_base"][1:], 5)

        # Build prompts
        final_prompt_no_response = prompt_1 + item["nhyps_base"][0] + prompt_2 + '\n' + '\n'.join(other_hypothesis) + prompt_3
        final_prompt = final_prompt_no_response + item["Caption"] + '<|endoftext|>'

        # Tokenize with truncation
        input_ids_no_response = tokenizer.encode(final_prompt_no_response, truncation=True, max_length=MAX_LENGTH)
        input_ids = tokenizer.encode(final_prompt, truncation=True, max_length=MAX_LENGTH)

        # Labels (-1 for no_response part)
        labels = [-1] * len(input_ids_no_response) + input_ids[len(input_ids_no_response):]

        # Add tensors
        new_dict['input_ids_no_response'] = torch.tensor(input_ids_no_response, dtype=torch.int32)
        new_dict['input_ids'] = torch.tensor(input_ids, dtype=torch.int32)
        new_dict['labels'] = torch.tensor(labels, dtype=torch.int32)

        # Store raw text
        new_dict['input'] = final_prompt
        new_dict['ground_truth'] = item["Caption"]

        # Noise embedding
        noise_embedding = calculate_noise_embedding(other_hypothesis)
        new_dict['emb_diff'] = torch.tensor(noise_embedding)

        # Speech embeddings (use dummy if not present)
        if "whisper_emb" in item:
            new_dict['clean_speech'] = item["whisper_emb"]
            new_dict['noisy_speech'] = item["whisper_emb"]
        else:
            dummy_emb = np.zeros(384, dtype=np.float32)  # 384 = MiniLM embedding size
            new_dict['clean_speech'] = dummy_emb
            new_dict['noisy_speech'] = dummy_emb

        # Mouth ROI (skip if not present)
        new_dict['mouthroi'] = item.get("Mouthroi", None)

        all_pt.append(new_dict)

    # Save .pt file
    torch.save(all_pt, output_path)
    print(f"✅ Saved {os.path.basename(output_path)} to {output_folder}")

# -------------------------------
# Convert both train & test
# -------------------------------
convert_json_to_pt(train_json_path, os.path.join(output_folder, "mciro_train_whisper_tiny.pt"))
convert_json_to_pt(test_json_path, os.path.join(output_folder, "mciro_test_whisper_tiny.pt"))
