import streamlit as st
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from exception import AtisException
import os,sys

# Loding all neessary files
model_filename = 'best_svm_model_ngram.joblib'
model = joblib.load(model_filename)


vectorizer_filename = 'tfidf_vectorizer_ngram.joblib'
tfidf_vectorizer = joblib.load(vectorizer_filename)


label_encoder_filename = 'label_encoder.joblib'
label_encoder = joblib.load(label_encoder_filename)


nltk.download('punkt')
nltk.download('stopwords')


def clean_text(text):
    try:
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word.isalnum()]
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        return ' '.join(tokens)
    except Exception as e:
        raise AtisException(e,sys)


def generate_response(intent):

    try:
        responses = {
            'atis_flight': "Here are the available flights...",
            'atis_airfare': "The airfare details are...",
            'atis_flight_time': "The flight times are...",
            'atis_aircraft': "The aircraft details are...",
            'atis_ground_service': "The ground service details are...",
            'atis_abbreviation': "The abbreviation details are..."
        }
        return responses.get(intent, "I'm sorry, I didn't understand that.")
    except Exception as e:
        raise AtisException(e,sys)

# Streamline code
st.title("Airline Travel Information Chatbot")

user_query = st.text_input("Enter your query:")

if st.button("Predict"):
    try:
        if user_query:

            cleaned_query = clean_text(user_query)
            
            query_vector = tfidf_vectorizer.transform([cleaned_query])
            
            predicted_label = model.predict(query_vector)[0]
            
            predicted_intent = label_encoder.inverse_transform([predicted_label])[0]
            
            response = generate_response(predicted_intent)
            
            st.write(f"Predicted Intent: {predicted_intent}")
            st.write(f"Response: {response}")
        else:
            st.write("Please enter a query.")
    except Exception as e:
        raise AtisException(e,sys)