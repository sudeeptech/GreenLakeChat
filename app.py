import streamlit as st
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -------------------------
# PAGE
# -------------------------
st.set_page_config(page_title="Local QA Assistant")
st.title("💬 Local Question Answer System (No API)")

# -------------------------
# LOAD QA DATA
# -------------------------
@st.cache_resource
def load_data():
    with open("qa_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = [item["question"] for item in data]
    answers = [item["answer"] for item in data]

    # Load local embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Convert questions → vectors
    embeddings = model.encode(questions)

    return model, questions, answers, embeddings


model, questions, answers, embeddings = load_data()

st.success("✅ Knowledge loaded")

# -------------------------
# USER INPUT
# -------------------------
query = st.text_input("Ask your question")

if query:

    # Convert user question to vector
    query_vector = model.encode([query])

    # Find similarity
    scores = cosine_similarity(query_vector, embeddings)[0]
    best_match = np.argmax(scores)
    confidence = scores[best_match]

    # Threshold check
    if confidence > 0.5:
        st.write("### Answer")
        st.write(answers[best_match])
        st.caption(f"Confidence: {round(confidence,2)}")
    else:
        st.write("I don't know")
