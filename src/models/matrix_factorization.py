import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import implicit
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

class SVDRecommender:
    """
    Truncated Singular Value Decomposition (SVD) for Explicit Ratings.
    Used for sparse matrix approximation.
    """
    def __init__(self, n_factors: int = 50):
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None
        self.user_to_idx = {}
        self.idx_to_item = {}
        self.global_mean = 0

    def fit(self, user_item_matrix: csr_matrix, user_to_idx: dict, item_to_idx: dict) -> None:
        logger.info(f"Running Truncated SVD with {self.n_factors} latent factors...")
        self.user_to_idx = user_to_idx
        self.idx_to_item = {idx: i for i, idx in item_to_idx.items()}
        
        # Convert to float for SVD
        matrix_float = user_item_matrix.astype(float)
        
        # svds is highly optimized for sparse matrices
        U, sigma, Vt = svds(matrix_float, k=self.n_factors)
        
        # Multiply U and Vt by the square root of singular values for balanced embeddings
        sigma = np.diag(np.sqrt(sigma))
        self.user_factors = np.dot(U, sigma)
        self.item_factors = np.dot(sigma, Vt).T
        logger.info("SVD Fit Complete.")

    def recommend(self, user_id: str, user_interacted_indices: set, k: int = 10) -> List[str]:
        if user_id not in self.user_to_idx:
            return []
            
        u_idx = self.user_to_idx[user_id]
        # Dot product of user embedding and all item embeddings
        scores = np.dot(self.item_factors, self.user_factors[u_idx])
        
        # Mask out already interacted items
        scores[list(user_interacted_indices)] = -np.inf
        
        top_k_indices = np.argsort(scores)[::-1][:k]
        return [self.idx_to_item[idx] for idx in top_k_indices]


class ALSRecommender:
    """
    Alternating Least Squares (ALS) for Implicit Feedback.
    Industry standard for clicks/views. Uses the `implicit` package.
    """
    def __init__(self, factors: int = 50, iterations: int = 15, regularization: float = 0.1):
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors, 
            iterations=iterations, 
            regularization=regularization,
            random_state=42
        )
        self.user_to_idx = {}
        self.idx_to_item = {}

    def fit(self, user_item_matrix: csr_matrix, user_to_idx: dict, item_to_idx: dict) -> None:
        logger.info("Training ALS Model...")
        self.user_to_idx = user_to_idx
        self.idx_to_item = {idx: i for i, idx in item_to_idx.items()}
        
        # The implicit library expects an item-user matrix for training
        item_user_matrix = user_item_matrix.T.tocsr()
        self.model.fit(item_user_matrix)
        logger.info("ALS Training Complete.")

    def get_item_embeddings(self) -> np.ndarray:
        return self.model.item_factors