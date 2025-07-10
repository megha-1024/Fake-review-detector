import streamlit as st
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Load model and BERT
model = joblib.load("xgb_model.pkl")
bert = SentenceTransformer('all-MiniLM-L6-v2')
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

# Streamlit UI
st.set_page_config(page_title="Fake Review Detector", page_icon="🔍")
st.title("🕵️ Fake Product Review Detector")
st.markdown("Enter a product review below to check if it's **Fake** or **Genuine**:")

review = st.text_area("✏️ Paste Review Here:")

if st.button("Detect Review"):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        # Feature Engineering
        review_length = len(review)
        word_count = len(review.split())
        sentiment = sid.polarity_scores(review)['compound']
        rating = 5  # Default rating input; can add a user input slider if needed

        # BERT Embedding
        emb = bert.encode([review])

        # Final input vector
        features = np.hstack((emb, [[review_length, word_count, sentiment, rating]]))

        # Prediction
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]  

        # Output
        if pred == 1:
            st.error(f"🚨 Prediction: **Fake Review**\n\nConfidence: {prob*100:.2f}%")
        else:
            st.success(f"✅ Prediction: **Genuine Review**\n\nConfidence: {(1 - prob)*100:.2f}%")
