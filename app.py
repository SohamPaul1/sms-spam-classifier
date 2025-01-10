import streamlit as st
import pickle as pkl
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# Load spaCy model
nlp = spacy.load('en_core_web_sm')


# Load Porter Stemmer equivalent in spaCy (lemmatizer)
def stem_word(token):
    return token.lemma_


tfidf = pkl.load(open('vectorizer.pkl', 'rb'))
model = pkl.load(open('model_mnb.pkl', 'rb'))

st.title("SMS SPAM CLASSIFIER")

# 1. Preprocess
def transform_text(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenize using spaCy
    doc = nlp(text)

    # Filter out stopwords, punctuation, and non-alphanumeric tokens
    res = [stem_word(token) for token in doc if token.is_alpha and token.text not in STOP_WORDS]

    return " ".join(res)

msg = st.text_area("ENTER YOUR SMS : ")

if st.button('Predict'):
    new_msg = transform_text(msg)

    # 2. Vectorize
    vector = tfidf.transform([new_msg])

    # 3. Predict
    result = model.predict(vector)[0]

    # Output
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")