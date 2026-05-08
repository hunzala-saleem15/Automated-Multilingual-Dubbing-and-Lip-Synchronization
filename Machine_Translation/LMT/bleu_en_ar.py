import sacrebleu

# Reference (ground truth Arabic)
with open("data/test1.ar", encoding="utf-8") as f:
    references = [f.readlines()]

# Base model output
with open("data/output_base_en_ar.ar", encoding="utf-8") as f:
    base_output = f.readlines()

# Fine-tuned model output
with open("data/output_en_ar_finetuned.txt", encoding="utf-8") as f:


    finetuned_output = f.readlines()

base_bleu = sacrebleu.corpus_bleu(base_output, references)
finetuned_bleu = sacrebleu.corpus_bleu(finetuned_output, references)

print("EN → AR Base Model BLEU:", base_bleu.score)
print("EN → AR Fine-Tuned Model BLEU:", finetuned_bleu.score)
