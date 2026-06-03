import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class ItemItemCF:
    """
    Memory-efficient Item-Item Collaborative Filtering using Sparse Matrices.
    """
    def __init__(self):
        self.item_sim_matrix = None
        self.user_item_matrix = None
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_item = {}

    def fit(self, df: pd.DataFrame, user_col: str = 'reviewerID', 
            item_col: str = 'asin', rating_col: str = 'overall') -> None:
        
        logger.info("Mapping IDs to integer indices...")
        self.user_to_idx = {u: i for i, u in enumerate(df[user_col].unique())}
        self.item_to_idx = {i: idx for idx, i in enumerate(df[item_col].unique())}
        self.idx_to_item = {idx: i for i, idx in self.item_to_idx.items()}
        
        row = df[user_col].map(self.user_to_idx).values
        col = df[item_col].map(self.item_to_idx).values
        data = df[rating_col].values
        
        logger.info("Constructing Sparse CSR Matrix...")
        self.user_item_matrix = csr_matrix((data, (row, col)), 
                                           shape=(len(self.user_to_idx), len(self.item_to_idx)))
        
        logger.info("Computing Item-Item Cosine Similarity...")
        # Transpose to compute similarity between items (columns), not users (rows)
        # We use sparse cosine_similarity which avoids dense memory blowups
        self.item_sim_matrix = cosine_similarity(self.user_item_matrix.T, dense_output=False)
        logger.info("Fit complete.")

    def recommend(self, user_id: str, k: int = 10) -> List[str]:
        """Generates recommendations via vector dot product."""
        if user_id not in self.user_to_idx:
            return [] # Cold start: Handle externally by returning Popular items
            
        u_idx = self.user_to_idx[user_id]
        
        # User's historical ratings vector (sparse)
        user_history_vector = self.user_item_matrix[u_idx]
        
        # Dot product: user_history * item_similarity
        # This calculates a score for all items based on items the user liked
        scores = user_history_vector.dot(self.item_sim_matrix).toarray().flatten()
        
        # Set scores of already interacted items to 0
        interacted_indices = user_history_vector.indices
        scores[interacted_indices] = 0.0
        
        # Get top K indices
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        # Map back to Item IDs
        return [self.idx_to_item[idx] for idx in top_k_indices if scores[idx] > 0]