
import streamlit as st
import pickle

# Load the saved vectorizer and model
with open("vectorizer.pkl", "rb") as f:
    cv = pickle.load(f)

with open("spam_classifier.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit app code
st.title("Spam Message Classifier")
st.write("Enter a message to classify if it's spam or ham")

# Text input for the message
user_input = st.text_input("Message")

if user_input:
    # Transform the input using the loaded vectorizer
    transformed_message = cv.transform([user_input]).toarray()
    prediction = model.predict(transformed_message)

    # Output prediction
    if prediction[0] == "spam":
        st.error("This message is classified as Spam!")
    else:
        st.success("This message is classified as Ham!")
