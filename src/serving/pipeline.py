import pandas as pd
import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)

class RecommendationService:
    """
    The Complete Industrial Recommendation Funnel.
    Retrieval -> Feature Hydration -> Ranking -> Re-Ranking (Business Logic)
    """
    def __init__(self, retriever, ranker, feature_store: pd.DataFrame):
        self.retriever = retriever  # TwoTower + FAISS
        self.ranker = ranker        # XGBoostRanker
        # In a real system, feature_store is a Redis cluster or AWS SageMaker Feature Store
        self.feature_store = feature_store 

    def retrieve_candidates(self, user_embedding: np.ndarray, top_k: int = 1000) -> List[str]:
        """Stage 1: Fast Vector Search"""
        logger.info(f"Stage 1: Retrieving {top_k} candidates via FAISS...")
        candidate_ids, _ = self.retriever.search(user_embedding, k=top_k)
        return candidate_ids

    def hydrate_features(self, user_id: str, candidate_ids: List[str]) -> pd.DataFrame:
        """
        Pulls real-time and historical features for the user and candidates.
        """
        logger.info(f"Hydrating features for {len(candidate_ids)} candidates...")
        # Simulate fetching item features from the Feature Store
        candidates_df = self.feature_store[self.feature_store['asin'].isin(candidate_ids)].copy()
        candidates_df['user_id'] = user_id
        
        # In production, we'd merge user features, context features, and cross features here.
        return candidates_df

    def rank_candidates(self, hydrated_candidates: pd.DataFrame, top_k: int = 100) -> pd.DataFrame:
        """Stage 2: Heavy Ranking Model"""
        logger.info(f"Stage 2: Ranking candidates using XGBoost...")
        ranked_df = self.ranker.predict_and_rank(
            hydrated_candidates, 
            features=self.ranker.features_names
        )
        return ranked_df.head(top_k)

    def rerank_business_logic(self, ranked_df: pd.DataFrame, top_k: int = 10) -> List[str]:
        """
        Stage 3: Re-ranking and Business Rules.
        """
        logger.info("Stage 3: Applying Business Rules...")
        
        # Rule 1: Remove out of stock (simulated)
        # ranked_df = ranked_df[ranked_df['in_stock'] == True]
        
        # Rule 2: Diversity - Don't show 10 items from the exact same sub-category
        final_recommendations = []
        seen_categories = set()
        
        for _, row in ranked_df.iterrows():
            # Let's pretend we have a 'category' column
            # if row['category'] not in seen_categories or len(final_recommendations) > 5:
            final_recommendations.append(row['asin'])
            # seen_categories.add(row['category'])
                
            if len(final_recommendations) == top_k:
                break
                
        return final_recommendations

    def serve_recommendations(self, user_id: str, user_embedding: np.ndarray) -> List[str]:
        """End-to-End Execution Pipeline."""
        # 1. Retrieval
        candidates = self.retrieve_candidates(user_embedding, top_k=1000)
        
        if not candidates:
            return [] # Fallback to Popularity
            
        # 2. Hydration
        hydrated_df = self.hydrate_features(user_id, candidates)
        
        # 3. Ranking
        ranked_df = self.rank_candidates(hydrated_df, top_k=100)
        
        # 4. Re-ranking
        final_recs = self.rerank_business_logic(ranked_df, top_k=10)
        
        return final_recs