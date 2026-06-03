import pandas as pd
import re
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from src.configs import PROCESSED_DATA_PATH, DF_SAVE_PATH, SIMILARITY_MATRIX_PATH, MODEL_DIR

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def train_and_save_model():
    """Reads data, calculates similarity matrix, and saves it to disk."""
    print("Loading data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df = df.reset_index(drop=True)
    
    print("Processing text...")
    df['tags'] = df['name'].apply(clean_text)
    
    print("Calculating TF-IDF and Similarity Matrix...")
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['tags'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    # Ensure the models directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("Saving models to disk...")
    joblib.dump(df, DF_SAVE_PATH)
    joblib.dump(cosine_sim, SIMILARITY_MATRIX_PATH)
    print("Model training and saving complete!")

def get_recommendations(product_name, top_n=5):
    """Loads the saved models and returns recommendations."""
    # Load the saved models
    df = joblib.load(DF_SAVE_PATH)
    cosine_sim = joblib.load(SIMILARITY_MATRIX_PATH)
    
    # Create the indices mapping
    indices = pd.Series(df.index, index=df['name']).drop_duplicates()
    
    if product_name not in indices:
        return None
    
    idx = indices[product_name]
    if type(idx) is pd.Series:
        idx = idx.iloc[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    
    product_indices = [i[0] for i in sim_scores]
    return df.iloc[product_indices][['name', 'image', 'ratings', 'discount_price', 'link']]

def get_trending_products(top_n=10):
    """Loads the saved dataframe and returns trending items."""
    df = joblib.load(DF_SAVE_PATH)
    min_ratings = 50
    trending = df[df['no_of_ratings'] >= min_ratings].copy()
    trending['popularity_score'] = trending['ratings'] * trending['no_of_ratings']
    trending = trending.sort_values(by='popularity_score', ascending=False)
    return trending[['name', 'image', 'ratings', 'discount_price', 'link']].head(top_n)