import sys
import os
import streamlit as st
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import get_recommendations, get_trending_products
from src.configs import DF_SAVE_PATH

st.set_page_config(page_title="E-Commerce Recommender", page_icon="🛒", layout="wide")

@st.cache_data
def load_product_names():
    df = joblib.load(DF_SAVE_PATH)
    return df['name'].dropna().tolist()

product_list = load_product_names()

st.title("🛒 E-Commerce Recommendation Engine")
st.markdown("A production-grade recommendation system built with Machine Learning.")

st.subheader("🔍 Find Similar Products")
selected_product = st.selectbox(
    "Search for a product you like:", 
    product_list, 
    index=0
)

if st.button("Recommend Similar Items"):
    with st.spinner("Our AI is searching for matches..."):
        recommendations = get_recommendations(selected_product, top_n=5)
        
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

st.subheader("🔥 Trending Products Right Now")
trending_df = get_trending_products(top_n=5)

t_cols = st.columns(5)
for idx, row in trending_df.reset_index().iterrows():
    with t_cols[idx]:
        st.image(row['image'], use_container_width=True)
        st.write(f"**{row['name'][:50]}...**")
        st.write(f"⭐ **{row['ratings']}**")
        st.write(f"💰 **₹{row['discount_price']}**")
        st.markdown(f"[View on Store]({row['link']})")