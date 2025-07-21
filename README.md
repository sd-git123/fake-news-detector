## 📰 Fake News Detector

A machine learning-powered web app built with Streamlit to classify whether a given news article is **real** or **fake**.

## 🚀 Features

- Input any news article or headline and get instant prediction
- Built with:
  - TF-IDF Vectorizer
  - PassiveAggressiveClassifier
- Displays model accuracy on test data
- Simple and interactive Streamlit UI

## 📊 Dataset

This project uses the open-source dataset from Kaggle:

- **Fake.csv** and **True.csv**: [Fake and Real News Dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

> ⚠️ Note: Due to file size, dataset files are not uploaded to this repository. Please download from the link above and place `Fake.csv` and `True.csv` in the project folder.

## 🛠️ Installation

1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/fake-news-detector.git
   cd fake-news-detector

Install dependencies:
pip install -r requirements.txt

Run the app:
streamlit run app.py

Requirements:
Python 3.8+
See requirements.txt for full list
