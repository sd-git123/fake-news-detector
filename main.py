import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load datasets
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Label data
fake['label'] = 0
real['label'] = 1

# Combine and shuffle
df = pd.concat([fake, real])
df = df[['text', 'label']].sample(frac=1).reset_index(drop=True)

# TF-IDF + train/test split
X = df['text']
y = df['label']
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vect = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Train model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train, y_train)

# Streamlit UI
st.title("📰 Fake News Detector")

st.write("Enter a news article or headline below:")

user_input = st.text_area("Paste your news text here", height=200)

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_vect = vectorizer.transform([user_input])
        result = model.predict(input_vect)[0]
        if result == 1:
            st.success("✅ This news appears to be **REAL**.")
        else:
            st.error("⚠️ This news appears to be **FAKE**.")

# Optional accuracy
st.caption(f"Model accuracy: {accuracy_score(y_test, model.predict(X_test)) * 100:.2f}%")
