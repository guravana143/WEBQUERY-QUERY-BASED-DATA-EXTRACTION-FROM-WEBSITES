import os
import requests
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import sqlite3
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage


def scrape_website(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        extracted_text = [paragraph.get_text() for paragraph in paragraphs]

        joined_text = " ".join(extracted_text)
        cleaned_text = re.sub(r'\s+', ' ', joined_text).strip()
        return cleaned_text
    except Exception as e:
        st.error(f"An error occurred while scraping the website: {e}")
        return None


def split_text(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return sentences


def vectorize_data(sentences):
    try:
        vectorizer = TfidfVectorizer()
        tfidf_vectors = vectorizer.fit_transform(sentences)
        return tfidf_vectors, vectorizer
    except Exception as e:
        st.error(f"An error occurred during vectorization: {e}")
        return None, None


def store_vectors_in_database(tfidf_vectors, sentences):
    try:
        with sqlite3.connect('vector_database.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''DROP TABLE IF EXISTS tfidf_vectors''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS tfidf_vectors (
                                id INTEGER PRIMARY KEY,
                                sentence TEXT,
                                vector BLOB
                            )''')
            for i, vector in enumerate(tfidf_vectors):
                sentence = sentences[i]
                vector_serialized = pickle.dumps(vector)
                cursor.execute('''INSERT INTO tfidf_vectors (sentence, vector)
                                  VALUES (?, ?)''', (sentence, vector_serialized))
        st.write("TF-IDF vectors stored in database successfully.")
    except Exception as e:
        st.error(f"An error occurred while storing vectors in the database: {e}")


def keyword_matching(question, sentences):
    matching_sentences = []
    for sentence in sentences:
        if re.search(r'\b{}\b'.format(re.escape(question)), sentence, re.IGNORECASE):
            matching_sentences.append(sentence)
    return matching_sentences


def semantic_search(question, vectorizer, threshold=0.20):
    try:
        question_vector = vectorizer.transform([question])
        with sqlite3.connect('vector_database.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''SELECT * FROM tfidf_vectors''')
            rows = cursor.fetchall()
            tfidf_vectors = [pickle.loads(row[2]) for row in rows]

        similarities = [cosine_similarity(question_vector, v)[0][0] for v in tfidf_vectors]
        max_similarity_index = np.argmax(similarities)
        max_similarity = similarities[max_similarity_index]

        print("max_similarity :",max_similarity)

        if max_similarity < threshold:
            return "Sorry, I don't know."
        else:
            most_similar_sentence = rows[max_similarity_index][1]
            return most_similar_sentence
    except Exception as e:
        st.error(f"An error occurred during semantic search: {e}")
        return None


def use_llm(question, max_response_length):
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")

        inputs = tokenizer.encode(question, return_tensors="pt", max_length=tokenizer.model_max_length, truncation=True)
        max_length = min(tokenizer.model_max_length, len(inputs[0]) + max_response_length)

        output = model.generate(inputs, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        return response
    except Exception as e:
        st.error(f"An error occurred during GPT-2 response generation: {e}")
        return None


def give_answer(answer):
    tokenizer = AutoTokenizer.from_pretrained("Mr-Vicky-01/Bart-Finetuned-conversational-summarization")
    model = AutoModelForSeq2SeqLM.from_pretrained("Mr-Vicky-01/Bart-Finetuned-conversational-summarization")

    def generate_summary(answer):
        inputs = tokenizer([answer], max_length=1024, return_tensors='pt', truncation=True)
        summary_ids = model.generate(inputs['input_ids'], max_new_tokens=100, do_sample=False)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    summary = generate_summary(answer)
    return summary


st.set_page_config(page_title="Webquery: Chat with Website", page_icon=":books:")
st.markdown(
    """
    <style>
    body {
        background-image: url('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRoRrZqwTS4lQYPOW3vpa3K5idRT0BGOQ43Hg&s');  
        color: blue;  
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Webquery: Chat with Website")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]

if "sentences" not in st.session_state:
    st.session_state.sentences = []

if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None

with st.sidebar:
    st.header("Website URL !!!")
    website_url = st.text_input("Enter the website URL .......")
    submit = st.button("Submit")

if submit:
    try:
        if os.path.exists('vector_database.db'):
            os.remove('vector_database.db')

        text = scrape_website(website_url)
        st.session_state.sentences = split_text(text)
        tfidf_vectors, st.session_state.vectorizer = vectorize_data(st.session_state.sentences)
        store_vectors_in_database(tfidf_vectors, st.session_state.sentences)
    except Exception as e:
        st.error(f"An error occurred: {e}")

user_query = st.text_input("Type your question here....")

if user_query and user_query.strip() and st.session_state.vectorizer:
    matching_sentences = keyword_matching(user_query, st.session_state.sentences)
    most_similar_sentence = semantic_search(user_query, st.session_state.vectorizer)

    if most_similar_sentence == "Sorry, I don't know.":
        answer = most_similar_sentence
    else:
        response = use_llm(most_similar_sentence, max_response_length=len(user_query.split()) * 5)
        answer = give_answer(response)

    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=answer))

def render_chat_message(message, icon):
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <img src="{icon}" alt="icon" style="width: 50px; height: 50px; margin-right: 1rem;">
            <div>{message}</div>
        </div>
        """, unsafe_allow_html=True
    )



human_icon_url = "data:image/png;base64,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"
ai_icon_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRdXaPfUob7UJWACFD-DKjSHuxCpJ0tg-uDmw&s"

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        render_chat_message(message.content, ai_icon_url)
    elif isinstance(message, HumanMessage):
        render_chat_message(message.content, human_icon_url)
