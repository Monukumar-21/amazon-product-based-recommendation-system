import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_amazon_products.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DF_SAVE_PATH = os.path.join(MODEL_DIR, 'df_model.joblib')
SIMILARITY_MATRIX_PATH = os.path.join(MODEL_DIR, 'cosine_sim.joblib')