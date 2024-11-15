import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Load the dataset
df = pd.read_csv("spam .csv")  # Ensure the correct path to your CSV file
y = df['Category']  # The target column, typically labeled as 'spam' or 'ham'
x = df['Message']  # The text messages to classify

# Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Initialize CountVectorizer and fit on training data
cv = CountVectorizer()
x_train = cv.fit_transform(x_train.values).toarray()
x_test = cv.transform(x_test.values).toarray()  # Transform x_test without re-fitting

# Initialize and train the Naive Bayes model
model = MultinomialNB()
model.fit(x_train, y_train)

# Evaluate model performance
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the vectorizer and model using pickle
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(cv, f)

with open("spam_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model and vectorizer saved successfully.")



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
