# Sentiment-Analysis
# Sentiment Analysis

## 📌 Overview
This project is a **Sentiment Analysis App** built with **Python, Streamlit, and Scikit-learn**. It classifies text as **positive or negative** using **Natural Language Processing (NLP)** techniques.

## 🚀 Features
- **Preprocesses text** by removing stopwords and special characters.
- **Uses TF-IDF Vectorization** to convert text into numerical features.
- **Trains a Naïve Bayes classifier** for sentiment detection.
- **Interactive Web UI** built with Streamlit.
- **Deployable on Streamlit Cloud or any cloud platform.**

## 📂 Project Structure
```
Sentiment Analysis/
│── model_training.py  # Trains the sentiment analysis model
│── app.py             # Streamlit web app
│── data_loader.py     # Loads and preprocesses data
│── requirements.txt   # List of dependencies
│── README.md          # Project documentation
```

## 🔧 Installation & Setup
### 1️⃣ Clone the repository
```sh
git clone https://github.com/AniketR10/Sentiment-Analysis.git
cd Sentiment-Analysis
```

### 2️⃣ Install dependencies
```sh
pip install -r streamlit
pandas
numpy
nltk
scikit-learn

```

### 3️⃣ Run the application
```sh
streamlit run app.py
```

## 📊 Model Details
- Uses **Naïve Bayes (MultinomialNB)** for classification.
- Trained on a dataset of labeled tweets (positive/negative).
- Uses **TF-IDF Vectorizer** to convert text to numerical form.


🚀 **Developed by [AniketR10](https://github.com/AniketR10)**

