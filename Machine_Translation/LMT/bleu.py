import sacrebleu

with open("data/test.es", encoding="utf-8") as f:
    references = [f.readlines()]

with open("data/output_base.es", encoding="utf-8") as f:
    base_output = f.readlines()

with open("data/output_finetuned.es", encoding="utf-8") as f:
    finetuned_output = f.readlines()

base_bleu = sacrebleu.corpus_bleu(base_output, references)
finetuned_bleu = sacrebleu.corpus_bleu(finetuned_output, references)

print("Base Model BLEU:", base_bleu.score)
print("Fine-Tuned Model BLEU:", finetuned_bleu.score)
