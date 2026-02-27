import matplotlib.pyplot as plt
import seaborn as sns


def plot_toxicity_distribution(df, out_png):
    plt.figure(figsize=(6, 4))
    sns.histplot(df["tox_score"], bins=20, kde=True)
    plt.xlabel("Toxicity score")
    plt.ylabel("Count")
    plt.title("Toxicity Score Distribution")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
