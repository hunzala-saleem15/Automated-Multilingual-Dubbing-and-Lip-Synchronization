import matplotlib.pyplot as plt

# Base paper BLEU (WMT News EN→ZH)
base_bleu = 85.20

# Fine-tuned model BLEU scores
comparisons = [
    ("Europarl EN→ES", 85.32),
    ("CoVoST-2 EN→AR", 3.50),
    ("OPUS-100 EN→HI", 28.20)
]

for name, bleu in comparisons:
    plt.figure()
    
    labels = ["Base Paper (EN→ZH)", name]
    scores = [base_bleu, bleu]
    
    plt.bar(labels, scores, color="#C3B1E1")  # Pastel Purple
    
    plt.ylabel("BLEU Score")
    plt.title(f"BLEU Comparison: {name} vs Base Paper")
    
    # Values on bars
    for i, score in enumerate(scores):
        plt.text(i, score + 1, f"{score}", ha="center", fontsize=10)
    
    plt.ylim(0, 100)
    plt.tight_layout()
    
    # Save for paper
    file_name = name.replace(" ", "_").replace("→", "to")
    plt.savefig(f"{file_name}_vs_base.png", dpi=300)
    
    plt.show()
