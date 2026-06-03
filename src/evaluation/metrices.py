import numpy as np
from typing import List

class RecommenderMetrics:
    """
    Standard Offline Evaluation Metrics for Recommendation Systems.
    """
    
    @staticmethod
    def precision_at_k(actual: List[str], predicted: List[str], k: int = 10) -> float:
        """Percentage of recommended items that are relevant."""
        if not actual or not predicted:
            return 0.0
        predicted_k = predicted[:k]
        hits = len(set(predicted_k).intersection(set(actual)))
        return hits / k

    @staticmethod
    def recall_at_k(actual: List[str], predicted: List[str], k: int = 10) -> float:
        """Percentage of relevant items that were successfully recommended."""
        if not actual:
            return 0.0
        predicted_k = predicted[:k]
        hits = len(set(predicted_k).intersection(set(actual)))
        return hits / len(actual)

    @staticmethod
    def hit_rate_at_k(actual: List[str], predicted: List[str], k: int = 10) -> int:
        """Binary metric: 1 if AT LEAST ONE relevant item is recommended, 0 otherwise."""
        if not actual or not predicted:
            return 0
        predicted_k = predicted[:k]
        return 1 if len(set(predicted_k).intersection(set(actual))) > 0 else 0