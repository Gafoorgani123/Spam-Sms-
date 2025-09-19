import pandas as pd
import numpy as np
import re
import string
import joblib
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the dataset
file_path =file_path = r"C:\Users\ANJANA ROHAN\Downloads\SpamSmsProj\spamraw.csv"
df = pd.read_csv(file_path)

# Convert labels: 'ham' -> 0, 'spam' -> 1
df['type'] = df['type'].map({'ham': 0, 'spam': 1})

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()  # Remove leading/trailing spaces
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Apply text cleaning with Lemmatization
df['cleaned_message'] = df['text'].apply(clean_text)

# Convert text messages into numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_message']).toarray()
y = df['type']

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naïve Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the trained model and vectorizer for later use
joblib.dump(model, "spam_sms_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully!")

# Function to predict new SMS messages
def predict_sms(text):
    model = joblib.load("spam_sms_model.pkl")  # Load saved model
    vectorizer = joblib.load("tfidf_vectorizer.pkl")  # Load saved vectorizer
    cleaned_text = clean_text(text)  # Preprocess input text
    vectorized_text = vectorizer.transform([cleaned_text]).toarray()  # Convert text to numerical format
    prediction = model.predict(vectorized_text)[0]  # Get prediction
    return "Spam" if prediction == 1 else "Ham (Not Spam)"

# Example: Predict new SMS messages
new_sms_list = [
    "Congratulations! You have won a free iPhone. Click the link now!",
    "Hey, how are you? Are we still meeting later?",
    "URGENT: Your account has been compromised. Please reset your password immediately.",
    "Let's catch up for coffee tomorrow!"
]

for sms in new_sms_list:
    print(f"Message: {sms} --> Prediction: {predict_sms(sms)}")
