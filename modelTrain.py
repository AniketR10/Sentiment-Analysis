import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from dataLoad import load_dataset

def train_model():
    data = load_dataset()
    X = data['processed_text']
    y = data['sentiment']

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    with open("sentiment_model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)
    with open("vectorizer.pkl", "wb") as vec_file:
        pickle.dump(vectorizer, vec_file)

if __name__ == "__main__":
    train_model()
