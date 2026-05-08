import matplotlib.pyplot as plt

# BLEU scores
models = ['Base Model', 'Fine-Tuned Model']
bleu_scores = [18.94, 3.50]

plt.figure(figsize=(6,4))
plt.bar(models, bleu_scores)
plt.ylabel('BLEU Score')
plt.title('EN → AR BLEU Score Comparison')

# value labels
for i, v in enumerate(bleu_scores):
    plt.text(i, v + 0.5, f"{v}", ha='center')

plt.tight_layout()

# save graph
plt.savefig("en_ar_bleu_comparison.png")
plt.show()
