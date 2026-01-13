import joblib

# Load trained objects
nb = joblib.load("nb_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
target_names = joblib.load("target_names.pkl")

def predict_topic(text):
    """
    Convert raw text to features and predict topic
    """
    X = vectorizer.transform([text])   # text → feature vector
    pred = nb.predict(X)[0]             # numeric label
    return target_names[pred]           # human-readable label


if __name__ == "__main__":
    print("=== Topic Classification App (Naive Bayes) ===")
    print("Type 'exit' to quit\n")

    while True:
        user_text = input("Enter text: ")

        if user_text.lower() == "exit":
            print("Exiting...")
            break

        topic = predict_topic(user_text)
        print(f"Predicted topic: {topic}\n")
