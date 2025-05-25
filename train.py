import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download NLTK resources
nltk.download("punkt_tab")
nltk.download("stopwords")

# Validate the data file
try:
    with open("intents.json", "r") as file:
        intents = json.load(file)
        if not intents.get("intents"):
            raise ValueError("The intents file is empty or improperly formatted.")
except Exception as e:
    print(f"Error loading intents file: {e}")
    exit()

# Preprocess the data
X, y = [], []
stop_words = set(stopwords.words("english"))

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        tokens = word_tokenize(pattern)
        filtered = [word.lower() for word in tokens if word.lower() not in stop_words]
        X.append(" ".join(filtered))
        y.append(intent["tag"])

# Convert labels to NumPy array
y = np.array(y)

# Convert text to numerical data using TF-IDF
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train the Naive Bayes model
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = classifier.predict(X_test)
print("\nModel Evaluation:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save the model and vectorizer
model_data = {"vectorizer": vectorizer, "classifier": classifier}
with open("nlp_model.pkl", "wb") as model_file:
    pickle.dump(model_data, model_file)

print("\nModel trained, evaluated, and saved successfully!")
