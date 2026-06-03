import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class FeaturePipeline:
    """
    Industrial-grade Feature Engineering Pipeline for E-Commerce Recommendation.
    Transforms raw interaction data into User, Item, and Contextual features.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Expects a DataFrame with at least:
        ['reviewerID', 'asin', 'overall', 'unixReviewTime']
        """
        self.df = df.copy()
        # Convert unix time to datetime efficiently
        if 'unixReviewTime' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['unixReviewTime'], unit='s')
        
        # Calculate a global 'current_date' to simulate Recency
        self.max_date = self.df['timestamp'].max()
        logger.info(f"Initialized FeaturePipeline with {len(self.df)} records.")

    def build_user_features(self) -> pd.DataFrame:
        """
        Extracts historical behavior for each user.
        Scalability Note: We use highly optimized Pandas GroupBy aggregations.
        """
        logger.info("Building User Features...")
        
        user_features = self.df.groupby('reviewerID').agg(
            user_interaction_count=('asin', 'count'),
            user_avg_rating_given=('overall', 'mean'),
            user_rating_std=('overall', 'std'),
            last_interaction_date=('timestamp', 'max')
        ).reset_index()

        # Fill NaN std deviations (for users with only 1 review) with 0
        user_features['user_rating_std'] = user_features['user_rating_std'].fillna(0.0)
        
        # Recency: Days since last interaction
        user_features['user_recency_days'] = (self.max_date - user_features['last_interaction_date']).dt.days
        user_features.drop(columns=['last_interaction_date'], inplace=True)
        
        # Engagement Score (log-scaled to handle Power Law distribution)
        user_features['user_engagement_score'] = np.log1p(user_features['user_interaction_count'])
        
        return user_features

    def build_item_features(self) -> pd.DataFrame:
        """
        Extracts popularity and quality metrics for each product.
        """
        logger.info("Building Item Features...")
        
        item_features = self.df.groupby('asin').agg(
            item_popularity_count=('reviewerID', 'count'),
            item_avg_rating=('overall', 'mean'),
            item_rating_std=('overall', 'std')
        ).reset_index()
        
        item_features['item_rating_std'] = item_features['item_rating_std'].fillna(0.0)
        
        # Bayesian Average (Industry standard for smoothing ratings)
        # Prevents items with one 5-star review from outranking items with 1000 4.8-star reviews
        global_avg = self.df['overall'].mean()
        m = 10 # Minimum votes needed to be considered 'credible'
        
        item_features['item_bayesian_rating'] = (
            (item_features['item_popularity_count'] * item_features['item_avg_rating']) + (m * global_avg)
        ) / (item_features['item_popularity_count'] + m)

        return item_features

    def build_time_features(self) -> pd.DataFrame:
        """
        Extracts contextual time features from the interaction.
        """
        logger.info("Building Contextual Time Features...")
        time_df = self.df[['reviewerID', 'asin', 'timestamp']].copy()
        
        # Fast vectorized datetime extractions
        time_df['day_of_week'] = time_df['timestamp'].dt.dayofweek
        time_df['month'] = time_df['timestamp'].dt.month
        time_df['is_weekend'] = time_df['day_of_week'].isin([5, 6]).astype(np.int8)
        
        return time_df.drop(columns=['timestamp'])

    def execute_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Runs the full pipeline and returns feature tables."""
        user_feats = self.build_user_features()
        item_feats = self.build_item_features()
        context_feats = self.build_time_features()
        
        logger.info("Feature Pipeline Execution Complete.")
        return user_feats, item_feats, context_feats

if __name__ == "__main__":
    # Example Usage:
    # df = pd.read_json("data/raw/Electronics_5.json", lines=True, nrows=10000)
    # pipeline = FeaturePipeline(df)
    # user_df, item_df, context_df = pipeline.execute_pipeline()
    pass