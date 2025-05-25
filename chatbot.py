import random
import pickle
import json

# Load the model using pickle
try:
    with open("nlp_model.pkl", "rb") as model_file:
        model_data = pickle.load(model_file)
    vectorizer = model_data["vectorizer"]
    classifier = model_data["classifier"]
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load intents file
try:
    with open("intents.json", "r") as file:
        intents = json.load(file)
except Exception as e:
    print(f"Error loading intents: {e}")
    exit()

# Chatbot loop
print("Chatbot is running! (Type 'quit' to exit)\n")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Goodbye!")
        break

    # Transform user input and predict
    input_vectorized = vectorizer.transform([user_input])
    tag = classifier.predict(input_vectorized)[0]

    # Find responses
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            response = random.choice(intent["responses"])
            print(f"Chatbot: {response}")
            break
