import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Debugging print statement
st.write("📌 Debug: Streamlit is running")

# Load trained model and vectorizer
try:
    st.write("📌 Debug: Loading model...")
    model = joblib.load("spam_sms_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    st.write("✅ Model loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to clean and preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Function to predict SMS messages
def predict_sms(text):
    cleaned_text = clean_text(text)
    vectorized_text = vectorizer.transform([cleaned_text]).toarray()
    prediction = model.predict(vectorized_text)[0]
    return "Spam" if prediction == 1 else "Ham (Not Spam)"

# Streamlit UI
st.title("📩 Spam SMS Detection App")
st.write("Enter an SMS message below to check if it's spam or not.")

# Debugging print statement
st.write("📌 Debug: UI Loaded Successfully")

# Input text box
user_input = st.text_area("Enter SMS message:", "")

if st.button("Predict"):
    st.write("📌 Debug: Predict Button Clicked")
    if user_input.strip():
        prediction = predict_sms(user_input)
        st.subheader("Prediction:")
        st.success(f"**{prediction}**")
    else:
        st.warning("⚠️ Please enter a message.")
