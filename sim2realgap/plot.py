import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import plot_matched_features


def plot_kl_matrix(KL, names, title="KL Divergence Between Distributions"):
    plt.figure(figsize=(8, 6))
    plt.imshow(KL, cmap='coolwarm')
    plt.colorbar(label='KL Divergence')
    plt.xticks(range(len(names)), names, rotation=30, ha='right')
    plt.yticks(range(len(names)), names)
    plt.xlabel("Compared to (Q)")
    plt.ylabel("Dataset (P)")
    for i in range(len(names)):
        for j in range(len(names)):
            plt.text(j, i, f"{KL[i,j]:.1f}", ha="center", va="center",
                     color="white", fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_histograms(data_dict, title, xlabel, bins=50):
    """
    Plot overlapping histograms.

    Parameters
    ----------
    data_dict : dict of {label: np.ndarray}
    title : str
    xlabel : str
    bins : int
    """
    plt.figure(figsize=(8, 4))
    for label, values in data_dict.items():
        plt.hist(values, bins=bins, alpha=0.5, label=label, density=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()