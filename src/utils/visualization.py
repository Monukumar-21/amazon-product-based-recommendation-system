import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import logging

logger = logging.getLogger(__name__)

def plot_item_embeddings(item_embeddings: np.ndarray, item_labels: List[str] = None, sample_size: int = 1000):
    """
    Reduces high-dimensional embeddings to 2D using t-SNE and plots them.
    """
    logger.info(f"Running t-SNE on {sample_size} item embeddings...")
    
    # Subsample for plotting speed and clarity
    idx = np.random.choice(item_embeddings.shape[0], min(sample_size, item_embeddings.shape[0]), replace=False)
    embeddings_subset = item_embeddings[idx]
    
    # t-SNE projects n_factors (e.g., 50D) down to 2D
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_subset)
    
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, c='dodgerblue', edgecolors='w', s=50)
    
    # Optional: annotate a few points
    if item_labels:
        labels_subset = [item_labels[i] for i in idx]
        for i in range(10): # Just label a few so it's not cluttered
            plt.annotate(labels_subset[i], (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                         fontsize=9, alpha=0.7)
            
    plt.title("t-SNE Visualization of Item Embeddings (Latent Space)", fontsize=16)
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.grid(True, alpha=0.3)
    plt.show()