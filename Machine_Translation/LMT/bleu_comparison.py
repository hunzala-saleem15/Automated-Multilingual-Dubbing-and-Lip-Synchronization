import matplotlib.pyplot as plt

color = "#C3B1E1"

# ==============================
# EN → ES (Europarl)
# ==============================
plt.figure(figsize=(5, 4))
labels = ["Base (1.7B)", "Fine-tuned"]
scores = [27.22, 85.32]

bars = plt.bar(labels, scores, color=color)
plt.ylabel("BLEU Score")
plt.title("EN → ES Translation Performance Comparison")
plt.ylim(0, 100)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1,
             f"{height}", ha='center')

plt.tight_layout()
plt.savefig("comparison_en_es.png", dpi=300)
plt.show()


# ==============================
# EN → AR (CoVoST-2)
# ==============================
plt.figure(figsize=(5, 4))
labels = ["Base (1.7B)", "Fine-tuned"]
scores = [21.73, 3.50]

bars = plt.bar(labels, scores, color=color)
plt.ylabel("BLEU Score")
plt.title("EN → AR Translation Performance Comparison")
plt.ylim(0, 100)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1,
             f"{height}", ha='center')

plt.tight_layout()
plt.savefig("comparison_en_ar.png", dpi=300)
plt.show()


# ==============================
# EN → ZH (WMT News)
# ==============================
plt.figure(figsize=(5, 4))
labels = ["Base (1.7B)", "Fine-tuned"]
scores = [47.16, 85.20]

bars = plt.bar(labels, scores, color=color)
plt.ylabel("BLEU Score")
plt.title("EN → ZH Translation Performance Comparison")
plt.ylim(0, 100)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1,
             f"{height}", ha='center')

plt.tight_layout()
plt.savefig("comparison_en_zh.png", dpi=300)
plt.show()


# ==============================
# EN → HI (OPUS100)
# ==============================
plt.figure(figsize=(5, 4))
labels = ["Base (1.7B)", "Fine-tuned"]
scores = [28.47, 28.20]

bars = plt.bar(labels, scores, color=color)
plt.ylabel("BLEU Score")
plt.title("EN → HI Translation Performance Comparison")
plt.ylim(0, 100)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1,
             f"{height}", ha='center')

plt.tight_layout()
plt.savefig("comparison_en_hi.png", dpi=300)
plt.show()
