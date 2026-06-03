import pandas as pd
import random
from typing import List, Dict

class PopularityRecommender:
    """
    Recommends the top K most popular items.
    Crucial fallback mechanism for production systems.
    """
    def __init__(self):
        self.popular_items: List[str] = []

    def fit(self, df: pd.DataFrame, item_col: str = 'asin') -> None:
        """Learns the most popular items based on interaction frequency."""
        # In a real scenario, we'd use the 'bayesian_rating' from Phase 3
        popularity_counts = df[item_col].value_counts().reset_index()
        popularity_counts.columns = [item_col, 'count']
        self.popular_items = popularity_counts[item_col].tolist()

    def recommend(self, user_history: List[str], k: int = 10) -> List[str]:
        """
        Recommends popular items, EXCLUDING items the user already interacted with.
        """
        recs = []
        for item in self.popular_items:
            if item not in user_history:
                recs.append(item)
            if len(recs) == k:
                break
        return recs

class RandomRecommender:
    """Absolute worst-case benchmark."""
    def __init__(self):
        self.catalog: List[str] = []

    def fit(self, df: pd.DataFrame, item_col: str = 'asin') -> None:
        self.catalog = df[item_col].unique().tolist()

    def recommend(self, user_history: List[str], k: int = 10) -> List[str]:
        available = list(set(self.catalog) - set(user_history))
        return random.sample(available, min(k, len(available)))