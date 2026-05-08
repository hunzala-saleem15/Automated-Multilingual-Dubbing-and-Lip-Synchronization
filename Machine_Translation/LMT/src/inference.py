import argparse
import logging
import tqdm
import time
import regex
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_MODEL = "NiuTrans/LMT-60-1.7B"

def is_whitespace(string):
    pattern = r'^[\s\p{C}]+$'
    return regex.match(pattern, string) is not None

def clean_pred(pred):
    pred = pred.strip()
    return "#" if is_whitespace(pred) else pred

def main():
    parser = argparse.ArgumentParser()

    # generation controls
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--length_penalty", type=float, default=1.0)

    # paths
    parser.add_argument("-m", "--model_path", type=str, required=True)
    parser.add_argument("-t", "--test_file", type=str, required=True)
    parser.add_argument("-s", "--hypo_file", type=str, required=True)

    # decoding params
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("🔹 Loading tokenizer (BASE MODEL)")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        padding_side="left",
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    print("🔹 Loading base model")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        trust_remote_code=True
    ).to(device)

    print("🔹 Loading LoRA adapter")
    model = PeftModel.from_pretrained(base_model, args.model_path)
    model.eval()

    # 🔹 Read input
    with open(args.test_file, encoding="utf-8") as f:
        sentences = [x.strip() for x in f if x.strip()]

    results = []
    start = time.time()

    for sent in tqdm.tqdm(sentences):
        prompt = f"English: {sent}\nHindi:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
    **inputs,
    max_new_tokens=args.max_new_tokens,
    num_beams=args.num_beams,
    repetition_penalty=args.repetition_penalty,
    length_penalty=args.length_penalty,
    do_sample=False,
    eos_token_id=tokenizer.eos_token_id,
)


        decoded = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],

            skip_special_tokens=True
        )

        decoded = decoded.split("Hindi:")[-1].strip()
        decoded = decoded.split("English:")[0].strip()
        results.append(clean_pred(decoded))

    elapsed = time.time() - start
    print(f"✅ Done | Time: {elapsed:.2f}s")

    with open(args.hypo_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results))

if __name__ == "__main__":
    main()
