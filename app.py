import os

import json
import random
import nltk
import numpy as np
import streamlit as st
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download NLTK resources
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('punkt_tab')


# Load intents
with open("intents.json", "r") as file:
    intents = json.load(file)

# Initialize NLP tools
stemmer = LancasterStemmer()
words = []
labels = []
docs_x = []
docs_y = []

# Process intents
for intent in intents:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(pattern)
        docs_y.append(intent["tag"])
    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = sorted(set([stemmer.stem(w.lower()) for w in words if w.isalnum()]))
labels = sorted(labels)

vectorizer = CountVectorizer(vocabulary=words)
x_train = vectorizer.transform(docs_x).toarray()
y_train = np.array([labels.index(y) for y in docs_y])

# Train model
model = MultinomialNB()
model.fit(x_train, y_train)

def classify(text):
    bow = vectorizer.transform([text]).toarray()
    pred = model.predict(bow)[0]
    return labels[pred]

def chatbot_response(text):
    tag = classify(text)
    for intent in intents:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Sorry, I don't understand."

# Streamlit UI
st.title("🤖 IntentMate - NLP Chatbot")
st.write("Chat with me below!")

user_input = st.text_input("You:")
if user_input:
    response = chatbot_response(user_input)
    st.text_area("Chatbot:", response, height=100)
