import xgboost as xgb
import pandas as pd
import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class XGBoostRanker:
    """
    Industrial Pairwise Ranking Model using XGBoost.
    Scores and sorts candidates retrieved by the First-Stage Retrieval model.
    """
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 6):
        # objective='rank:pairwise' minimizes pairwise loss (BPR-like)
        self.model = xgb.XGBRanker(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            objective='rank:pairwise',
            eval_metric='ndcg',
            tree_method='hist', # Highly optimized for speed
            random_state=42
        )
        self.features_names: List[str] = []

    def fit(self, df: pd.DataFrame, group_col: str, target_col: str, features: List[str]) -> None:
        """
        Trains the Ranker. 
        Crucial: XGBRanker requires data to be sorted by the group column (Query/User ID).
        """
        logger.info("Training XGBoost Pairwise Ranker...")
        self.features_names = features
        
        # XGBoost requires groups (users/queries) to be contiguous
        df_sorted = df.sort_values(by=group_col)
        
        X = df_sorted[self.features_names]
        y = df_sorted[target_col]
        
        # 'qid' (Query ID) tells XGBoost which items belong to the same user session
        # so it knows which items to compare against each other pairwise.
        qids = df_sorted[group_col]

        self.model.fit(X, y, qid=qids, verbose=True)
        logger.info("Ranker training complete.")

    def predict_and_rank(self, candidates_df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Scores candidate items and sorts them by predicted relevance.
        """
        X_infer = candidates_df[features]
        
        # Predict pairwise scores
        candidates_df['rank_score'] = self.model.predict(X_infer)
        
        # Sort descending by score
        return candidates_df.sort_values(by='rank_score', ascending=False).reset_index(drop=True)