import os
import re
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Page configuration
st.set_page_config(page_title="E-Commerce Recommender", page_icon="🛒", layout="wide")

# --- CORE ENGINE (COMPUTE ON THE FLY) ---
def clean_text(text):
    if not isinstance(text, str): return ""
    return re.sub(r'[^a-z0-9\s]', '', text.lower())

@st.cache_data
def load_data():
    """Loads the CSV and prepares the tags"""
    # Dynamically find the path to the processed data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'processed', 'cleaned_amazon_products.csv')
    
    df = pd.read_csv(data_path)
    df = df.reset_index(drop=True)
    df['tags'] = df['name'].apply(clean_text)
    return df

@st.cache_resource
def compute_similarity(_df):
    """Calculates the 580MB matrix instantly and halves its size in RAM"""
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(_df['tags'])
    
    # Calculate similarity and convert to float32 to save RAM!
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix).astype(np.float32)
    return cosine_sim

# --- RECOMMENDATION LOGIC ---
def get_recommendations(product_name, df, cosine_sim, top_n=5):
    indices = pd.Series(df.index, index=df['name']).drop_duplicates()
    if product_name not in indices: return None
    
    idx = indices[product_name]
    if type(idx) is pd.Series: idx = idx.iloc[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    
    product_indices = [i[0] for i in sim_scores]
    return df.iloc[product_indices][['name', 'image', 'ratings', 'discount_price', 'link']]

def get_trending_products(df, top_n=10):
    min_ratings = 50
    trending = df[df['no_of_ratings'] >= min_ratings].copy()
    trending['popularity_score'] = trending['ratings'] * trending['no_of_ratings']
    trending = trending.sort_values(by='popularity_score', ascending=False)
    return trending[['name', 'image', 'ratings', 'discount_price', 'link']].head(top_n)

# --- BOOT UP THE APP ---
with st.spinner("Loading AI Engine... (This takes a few seconds)"):
    df = load_data()
    cosine_sim = compute_similarity(df)
    product_list = df['name'].dropna().tolist()

# --- Main App UI ---
st.title("🛒 E-Commerce Recommendation Engine")
st.markdown("A production-grade recommendation system built with Machine Learning.")

# --- Section 1: AI Recommendations (Content-Based) ---
st.subheader("🔍 Find Similar Products")
selected_product = st.selectbox("Search for a product you like:", product_list, index=0)

if st.button("Recommend Similar Items"):
    with st.spinner("Our AI is searching for matches..."):
        recommendations = get_recommendations(selected_product, df, cosine_sim, top_n=5)
        
        if recommendations is not None and not recommendations.empty:
            st.success("Here are your recommendations!")
            cols = st.columns(5)
            for idx, row in recommendations.reset_index().iterrows():
                with cols[idx]:
                    st.image(row['image'], use_container_width=True)
                    st.write(f"**{row['name'][:50]}...**")
                    st.write(f"⭐ **{row['ratings']}**")
                    st.write(f"💰 **₹{row['discount_price']}**")
                    st.markdown(f"[View on Store]({row['link']})")
        else:
            st.warning("Could not find recommendations for this product.")

st.markdown("---")

# --- Section 2: Trending Products (Popularity Baseline) ---
st.subheader("🔥 Trending Products Right Now")
trending_df = get_trending_products(df, top_n=5)

t_cols = st.columns(5)
for idx, row in trending_df.reset_index().iterrows():
    with t_cols[idx]:
        st.image(row['image'], use_container_width=True)
        st.write(f"**{row['name'][:50]}...**")
        st.write(f"⭐ **{row['ratings']}**")
        st.write(f"💰 **₹{row['discount_price']}**")
        st.markdown(f"[View on Store]({row['link']})")