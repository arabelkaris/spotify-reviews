import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model and tokenizer
model = load_model('my_model.keras')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# App title
st.title("Spotify Review Sentiment Analysis")
st.write("Upload a text review to analyse its sentiment (Positive/Negative).")

# Input text for analysis
user_input = st.text_area("Enter a Spotify review:")

# Preprocessing and prediction
if st.button("Analyse Sentiment"):
    if user_input.strip():
        # Preprocess and tokenize input
        seq = tokenizer.texts_to_sequences([user_input])
        padded_seq = pad_sequences(seq, maxlen=100)  # Ensure the same length as training

        # Predict
        prediction = model.predict(padded_seq)
        sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"

        # Display result
        st.write(f"**Sentiment Analysis Result:** {sentiment}")
        st.write(f"Prediction Score: {prediction[0][0]:.2f}")
    else:
        st.warning("Please enter a review to analyse.")

# Footer
st.write("Developed as part of the Spotify Review AI/DS Project.")
