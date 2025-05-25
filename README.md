FlightBookingChatbot
Overview
FlightBookingChatbot is a Python-based, NLP-powered chatbot that simplifies flight ticket booking through natural language interactions. Using a Naive Bayes classifier and TF-IDF vectorization, it processes user queries to handle greetings, ticket bookings, price inquiries, payment methods, confirmations, and cancellations.
Features

Intent Recognition: Classifies user inputs using a trained Naive Bayes model.
Flight Booking: Handles booking requests and provides sample flight prices.
Payment Support: Lists payment options and confirms payments.
Cancellation: Processes booking cancellation requests.
Extensible: Modular design for easy integration with flight APIs or databases.

Install dependencies:  pip install -r requirements.txt

Required packages: nltk, scikit-learn, numpy
Download NLTK resources:  import nltk
nltk.download("punkt_tab")
nltk.download("stopwords")


Train the model:  python train.py


Run the chatbot:  python chatbot.py

Usage

Run chatbot.py and interact with the chatbot via the console.
Type queries like "Book a ticket to London" or "What are the payment options?".
Type quit to exit.

Project Structure

intents.json: Defines intents and responses for the chatbot.
train.py: Trains the Naive Bayes model using TF-IDF vectorization.
flights.csv: Sample dataset with flight details.
chatbot.py: Main script for running the chatbot.
nlp_model.pkl: Saved model and vectorizer.

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a branch: git checkout -b feature/your-feature.
Commit changes: git commit -m 'Add your feature'.
Push to the branch: git push origin feature/your-feature.
Open a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions or feedback, open an issue or contact [your email or GitHub handle].
