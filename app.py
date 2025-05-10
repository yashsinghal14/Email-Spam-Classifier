import streamlit as st
import pickle
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from datetime import datetime

# Ensure required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Text transformation function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

# Load vectorizer and model
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please upload the necessary files.")
    st.stop()

# App styling
st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #dceefb, #f3e5f5);
        }
        .main {
            background-color: white;
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            max-width: 800px;
            margin: 0 auto;
        }
        .title {
            font-size: 2.5rem;
            color: #333;
            text-align: center;
            margin-bottom: 1rem;
        }
        .result {
            font-size: 1.5rem;
            font-weight: bold;
            text-align: center;
            margin-top: 1.5rem;
        }
        .history {
            font-size: 1rem;
            margin-top: 2rem;
            color: #444;
        }
        .log-table {
            width: 100%;
            margin-top: 20px;
            border-collapse: collapse;
        }
        .log-table th, .log-table td {
            padding: 10px;
            text-align: center;
            border: 1px solid #ddd;
        }
        .log-table th {
            background-color: #f4f4f4;
        }
        .button {
            display: flex;
            justify-content: center;
            gap: 10px;
        }
        .input-box {
            margin-bottom: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# Add logo
st.image("email.png", width=500)

# Title
st.markdown("<div class='title'>📩 Email/SMS Spam Classifier</div>", unsafe_allow_html=True)

# Input area for message
input_sms = st.text_area("✉️ Enter the message", height=150, placeholder="Type your message here...")

# Clear input button
clear_button = st.button("❌ Clear Input")

if clear_button:
    st.experimental_rerun()

# Prediction logic
if st.button('🔍 Predict'):
    if not input_sms.strip():
        st.error("Please enter a message before predicting.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])

        try:
            result = model.predict(vector_input)[0]
            confidence = model.predict_proba(vector_input).max()
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.stop()

        # Display result
        if result == 1:
            st.markdown(f"<div class='result' style='color:red;'>🚨 Spam Detected! <br> Confidence: {confidence*100:.2f}%</div>", unsafe_allow_html=True)
            prediction_label = "Spam"
        else:
            st.markdown(f"<div class='result' style='color:green;'>✅ Not Spam <br> Confidence: {confidence*100:.2f}%</div>", unsafe_allow_html=True)
            prediction_label = "Not Spam"

        # Log the prediction
        log = {
            "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "Message": [input_sms],
            "Prediction": [prediction_label],
            "Confidence": [round(confidence, 4)]
        }
        log_df = pd.DataFrame(log)

        # Save the log
        try:
            existing_log = pd.read_csv("predictions.csv")
            updated_log = pd.concat([existing_log, log_df], ignore_index=True)
        except FileNotFoundError:
            updated_log = log_df

        updated_log.to_csv("predictions.csv", index=False)

        # Display prediction history
        st.markdown("<div class='history'>📜 Prediction History</div>", unsafe_allow_html=True)
        st.table(updated_log[['Timestamp', 'Message', 'Prediction', 'Confidence']].tail(5))

# Footer with a custom message
st.markdown("<div class='footer' style='text-align: center; padding-top: 20px;'>Built with ❤️ by [Your Name]</div>", unsafe_allow_html=True)
