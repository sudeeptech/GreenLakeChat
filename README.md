# ------------------------------
# GreenLake Assistance App
# ------------------------------

import streamlit as st                     # Import Streamlit for building the web app interface
import json                                # Import JSON module to read Q&A data from a file
import numpy as np                         # Import NumPy for numerical operations (vectors, arrays)
from sentence_transformers import SentenceTransformer  # Import transformer model to embed text
from sklearn.metrics.pairwise import cosine_similarity # Import cosine similarity function to match embeddings

# ------------------------------
# PAGE CONFIGURATION
# ------------------------------
st.set_page_config(page_title="GreenLake Assistance")  # Sets the browser tab title
st.title("💬GreenLake Assistance")                     # Displays the main title on the app page

# ------------------------------
# RELOAD BUTTON (for updated JSON)
# ------------------------------
if st.button("🔄 Reload document"):    # Creates a clickable button labeled 'Reload document'
    st.cache_resource.clear()          # Clears cached data/model so updated JSON can be reloaded
    st.rerun()                         # Re-runs the app immediately after clearing cache

# ------------------------------
# LOAD QA DATA AND MODEL
# ------------------------------
@st.cache_resource(show_spinner=True)  # Decorator: caches output so the model/data load only once
def load_data():
    # Load Q&A JSON
    try:
        with open("qa_data.json", "r", encoding="utf-8") as f:  # Open JSON file safely
            data = json.load(f)                                  # Parse JSON into Python list of dictionaries
    except FileNotFoundError:
        st.error("❌ qa_data.json file not found")  # Show error in app if file missing
        st.stop()                                  # Stop execution if file not found
    except Exception as e:
        st.error(f"Error loading data: {e}")      # Show any other error while loading
        st.stop()                                  # Stop execution on error

    # Extract questions and answers separately
    questions = [item["question"] for item in data]  # List of all questions from JSON
    answers = [item["answer"] for item in data]     # List of all answers from JSON

    # Load embedding model (downloads once, then cached)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # Load pretrained embedding model

    # Convert all questions to embeddings (vectors)
    embeddings = model.encode(questions, convert_to_numpy=True)  # Embeds each question to a numerical vector

    return model, questions, answers, embeddings  # Return the model, questions, answers, and embeddings

# Load data once (cached)
model, questions, answers, embeddings = load_data()  # Call load_data() to get everything
st.success("✅ Knowledge Base Loaded")               # Display success message when KB is loaded

# ------------------------------
# USER INPUT
# ------------------------------
query = st.text_input("Ask your question")  # Create a text box where user can type a question

# ------------------------------
# PROCESS QUERY
# ------------------------------
if query:  # Only run if the user has entered a question
    # Convert user question → vector embedding
    query_vector = model.encode([query], convert_to_numpy=True)  # Embed user query into vector

    # Compute cosine similarity with all stored question embeddings
    scores = cosine_similarity(query_vector, embeddings)[0]       # Compute similarity between query and all questions

    # Find index of the highest similarity
    best_match_index = np.argmax(scores)  # Find which stored question is most similar

    # Check if similarity is above threshold
    if scores[best_match_index] > 0.5:    # Only return answer if similarity > 0.5
        st.write("### ✅ Answer")         # Header for answer
        st.write(answers[best_match_index])  # Display the most similar answer
    else:
        st.write("I don't know")          # Return default if no good match found
