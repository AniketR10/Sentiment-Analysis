import streamlit as st
import pickle
from dataLoad import preprocess_text

# Load trained model and vectorizer
with open("sentiment_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

st.title("Sentiment Analysis App")
st.write("Enter a text below to analyze its sentiment.")

user_input = st.text_area("Enter text:")

if st.button("Analyze Sentiment"):
    input_processed = preprocess_text(user_input)
    input_vectorized = vectorizer.transform([input_processed])
    prediction = model.predict(input_vectorized)[0]
    
    sentiment_map = {0: "Negative 😞", 1: "Neutral 😐", 2: "Positive 😊"}
    st.write(f"Prediction: {sentiment_map[prediction]}")
