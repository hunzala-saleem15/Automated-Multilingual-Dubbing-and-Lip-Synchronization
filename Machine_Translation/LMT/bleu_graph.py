import matplotlib.pyplot as plt

models = ["Base Model", "Fine-Tuned Model"]
bleu_scores = [53.99, 85.32]

plt.figure()
plt.bar(models, bleu_scores)
plt.xlabel("Model")
plt.ylabel("BLEU Score")
plt.title("BLEU Score Comparison (English → Spanish)")
plt.ylim(0, 100)

for i, score in enumerate(bleu_scores):
    plt.text(i, score + 1, f"{score:.2f}", ha="center")

plt.show()
