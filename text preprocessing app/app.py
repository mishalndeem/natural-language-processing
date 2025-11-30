# NLP Frontend App using Streamlit
# Save this file as `app.py` and run with: streamlit run app.py
# Requirements (install with pip):
# pip install streamlit scikit-learn nltk pandas numpy

import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure NLTK resources are available
try:
    _ = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# ---------- Text cleaning utilities ----------
DEFAULT_SAMPLE = (
    "Natural Language Processing (NLP) is a field of AI that gives machines the ability to read, "
    "understand and derive meaning from human languages. This example shows simple cleaning steps: "
    "lowercasing, removing numbers & punctuation, removing stopwords, tokenization and stemming."
)

ps = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))

def to_lowercase(text: str) -> str:
    return text.lower()

def remove_numbers(text: str) -> str:
    return re.sub(r"\d+", "", text)

def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_extra_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text: str) -> list:
    return re.findall(r"[A-Za-z]+", text)

def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in STOPWORDS]

def stem_tokens(tokens: list) -> list:
    return [ps.stem(t) for t in tokens]

def clean_text(text: str, *, do_lower=True, do_numbers=True, do_punct=True,
               do_stopwords=True, do_stem=False) -> dict:
    steps = {}
    s = text
    steps['original'] = s
    if do_lower:
        s = to_lowercase(s)
        steps['lowercase'] = s
    if do_numbers:
        s = remove_numbers(s)
        steps['no_numbers'] = s
    if do_punct:
        s = remove_punctuation(s)
        steps['no_punct'] = s
    s = remove_extra_whitespace(s)
    steps['trimmed'] = s
    tokens = tokenize(s)
    steps['tokens'] = tokens
    if do_stopwords:
        tokens = remove_stopwords(tokens)
        steps['no_stopwords'] = tokens
    if do_stem:
        tokens = stem_tokens(tokens)
        steps['stemmed'] = tokens
    steps['final_tokens'] = tokens
    steps['final_text'] = ' '.join(tokens)
    return steps

# ---------- TF-IDF utilities ----------
def compute_tfidf(docs: list, max_features: int = None, ngram_range=(1,1)) -> (pd.DataFrame, TfidfVectorizer):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vectorizer.fit_transform(docs)
    df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    return df, vectorizer

# ---------- Streamlit UI ----------
def app():
    st.set_page_config(page_title='NLP Frontend (Python)', layout='wide')
    st.title('NLP Frontend App — Text Processing')

    st.sidebar.header('Input')
    input_mode = st.sidebar.radio('Input mode', ['Paste / Type text', 'Upload text file (.txt)', 'Sample text'])

    text = ''
    if input_mode == 'Paste / Type text':
        text = st.sidebar.text_area('Enter text (multiple documents separated by a blank line):', height=200)
    elif input_mode == 'Upload text file (.txt)':
        uploaded_file = st.sidebar.file_uploader('Upload a .txt file', type=['txt'])
        if uploaded_file is not None:
            text = uploaded_file.read().decode('utf-8', errors='replace')
    else:
        text = st.sidebar.text_area('Sample text (editable):', value=DEFAULT_SAMPLE, height=200)

    if isinstance(text, str) and text.strip() == '':
        st.info('Enter or upload some text on the left to get started.')
        st.stop()

    # Split into documents
    raw_docs = [d.strip() for d in re.split(r"\n\s*\n", text) if d.strip()]

    # Ask the user what they want to do with the text
    st.header('What do you want to do with the text?')
    choice = st.radio("Select an option:", [
        "Clean the text",
        "Compute TF-IDF",
        "Show Vocabulary & Frequencies",
        "Do all of the above"
    ])

    # Cleaning
    if choice in ("Clean the text", "Do all of the above"):
        st.sidebar.header('Cleaning options')
        do_lower = st.sidebar.checkbox('Lowercase', value=True)
        do_numbers = st.sidebar.checkbox('Remove numbers', value=True)
        do_punct = st.sidebar.checkbox('Remove punctuation', value=True)
        do_stopwords = st.sidebar.checkbox('Remove stopwords', value=True)
        do_stem = st.sidebar.checkbox('Stemming (Porter)', value=False)

        cleaned_docs = []
        cleaning_steps = []
        for doc in raw_docs:
            steps = clean_text(doc, do_lower=do_lower, do_numbers=do_numbers, do_punct=do_punct,
                               do_stopwords=do_stopwords, do_stem=do_stem)
            cleaned_docs.append(steps['final_text'])
            cleaning_steps.append(steps)

        st.header('Cleaning Results')
        for i, steps in enumerate(cleaning_steps, 1):
            with st.expander(f'Document {i} — cleaning steps'):
                st.write('**Original**')
                st.write(steps['original'])
                st.write('**Final cleaned text**')
                st.write(steps['final_text'])

    # TF-IDF
    if choice in ("Compute TF-IDF", "Do all of the above"):
        st.sidebar.header('TF‑IDF options')
        max_features = st.sidebar.number_input('Max features (0 = no limit)', min_value=0, value=0, step=50)
        ngram_min = st.sidebar.number_input('ngram min', min_value=1, value=1, step=1)
        ngram_max = st.sidebar.number_input('ngram max', min_value=1, value=1, step=1)

        if st.button('Compute TF‑IDF'):
            mf = None if max_features == 0 else int(max_features)
            df_tfidf, _ = compute_tfidf(raw_docs, max_features=mf, ngram_range=(int(ngram_min), int(ngram_max)))
            st.dataframe(df_tfidf.T)

    # Vocabulary
    if choice in ("Show Vocabulary & Frequencies", "Do all of the above"):
        vocab = {}
        for doc in raw_docs:
            tokens = tokenize(remove_punctuation(to_lowercase(doc)))
            for t in tokens:
                vocab[t] = vocab.get(t, 0) + 1
        vocab_df = pd.DataFrame(sorted(vocab.items(), key=lambda x: x[1], reverse=True), columns=['term','freq'])
        st.header('Vocabulary & Frequencies')
        st.dataframe(vocab_df)

if __name__ == '__main__':
    app()
