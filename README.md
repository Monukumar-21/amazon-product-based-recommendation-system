# 🛒 E-Commerce Recommendation Engine

An end-to-end, production-ready Recommendation System built for E-commerce platforms. This project solves the "Cold Start" problem for new users using a **Popularity-Based** model and provides highly relevant product suggestions using a **Content-Based Filtering** (NLP) model.

## 🌟 Features
* **Trending Products (Cold Start Solution):** A baseline popularity model that ranks items based on a weighted mathematical formula (`ratings * number_of_ratings`).
* **AI-Powered Similar Products:** A Content-Based recommendation engine that uses **TF-IDF Vectorization** and **Cosine Similarity** to process product names and suggest visually and contextually similar items.
* **Production-Grade Architecture:** Clean modular code separating data ingestion, model training, and the web app interface. Models are serialized (`.joblib`) for lightning-fast frontend inference.
* **Interactive Web App:** A highly responsive frontend built with Streamlit, rendering live product images, prices, and direct store links.

## 🏗️ Project Structure
├── app/                  # Frontend Web Application
│   └── main.py           # Streamlit UI
├── data/                 # Ignored in git (Raw and cleaned datasets)
├── models/               # Ignored in git (Serialized joblib matrices)
├── notebooks/            # Jupyter notebooks for EDA and prototyping
├── src/                  # Production Source Code
│   ├── config.py         # Global variables and paths
│   ├── data_loader.py    # Data cleaning logic
│   └── model.py          # TF-IDF Training & Inference logic
├── requirements.txt      # Project dependencies
└── README.md
## working link
https://amazon-electronics-recommendation-system.streamlit.app/