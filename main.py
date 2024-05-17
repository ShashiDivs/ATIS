import streamlit as st
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Load the trained SVM model
model_filename = 'best_svm_model_ngram.joblib'
model = joblib.load(model_filename)

# Load the TF-IDF vectorizer
vectorizer_filename = 'tfidf_vectorizer_ngram.joblib'
tfidf_vectorizer = joblib.load(vectorizer_filename)

# Load the label encoder
label_encoder_filename = 'label_encoder.joblib'
label_encoder = joblib.load(label_encoder_filename)

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Define a function for text cleaning
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove punctuation
    tokens = [word for word in tokens if word.isalnum()]
    # Remove stop words
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Define a function to generate responses based on intent
def generate_response(intent):
    responses = {
        'atis_flight': "Here are the available flights...",
        'atis_airfare': "The airfare details are...",
        'atis_flight_time': "The flight times are...",
        # Add more intents and responses as needed
    }
    return responses.get(intent, "I'm sorry, I didn't understand that.")

# Streamlit app
st.title("Airline Travel Information Chatbot")

user_query = st.text_input("Enter your query:")

if st.button("Predict"):
    if user_query:
        # Clean the user query
        cleaned_query = clean_text(user_query)
        
        # Vectorize the cleaned query
        query_vector = tfidf_vectorizer.transform([cleaned_query])
        
        # Predict the intent label
        predicted_label = model.predict(query_vector)[0]
        
        # Map the label to the intent name
        predicted_intent = label_encoder.inverse_transform([predicted_label])[0]
        
        # Generate a response based on the predicted intent
        response = generate_response(predicted_intent)
        
        st.write(f"Predicted Intent: {predicted_intent}")
        st.write(f"Response: {response}")
    else:
        st.write("Please enter a query.")