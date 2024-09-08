import os
import sqlite3
import streamlit as st
import fitz  # PyMuPDF
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline  

qa_model = pipeline("text2text-generation", model="google/flan-t5-large")


DB_PATH = 'chat_history.db'

# Function to initialize the SQLite database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            user_question TEXT,
            bot_answer TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Function to save a chat to the database
def save_to_db(user_id, user_question, bot_answer):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO chat_history (user_id, user_question, bot_answer) VALUES (?, ?, ?)', (user_id, user_question, bot_answer))
    conn.commit()
    conn.close()

# Function to load chat history from the database
def load_from_db(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT user_question, bot_answer FROM chat_history WHERE user_id = ?', (user_id,))
    rows = c.fetchall()
    conn.close()
    return rows

# Function to read and extract text from a PDF file
def read_pdf(file):
    text = ""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Function to split text into chunks
def split_text_into_chunks(text, chunk_size=1000, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Function to create TF-IDF embeddings for text chunks
def create_embeddings(chunks):
    vectorizer = TfidfVectorizer(stop_words='english')
    embeddings = vectorizer.fit_transform(chunks)
    return vectorizer, embeddings

# Function to retrieve relevant text chunks based on a query
def retrieve_relevant_chunks(query, vectorizer, embeddings, chunks, top_k=3):
    query_embedding = vectorizer.transform([query])
    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# Function to detect user intent
def detect_intent(user_input):
    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    if user_input.lower().strip() in greetings:
        return "greeting"
    else:
        return "question"

# Function to handle greetings
def handle_greeting():
    return "Hello! How can I assist you today?"

# Function to generate a response using the Hugging Face model for detailed answers
def generate_response(query, context):
    # Use the `text2text-generation` pipeline for generating more elaborate answers
    prompt = f"Question: {query}\nContext: {context}\n\nProvide a detailed and comprehensive answer of atleast 500 characters. use proper conversational english to provide full polite answers:"
    response = qa_model(prompt, max_length=512, num_return_sequences=1)
    return response[0]['generated_text']

# Main function for the Streamlit app
def main():
    st.set_page_config(page_title="RAG based AI ChatBot model", page_icon="🤖", layout="wide")

    # Custom CSS for styling
    st.markdown("""
    <style>
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css');
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
    }
    .chat-message.user {
        background-color: #2b313e
    }
    .chat-message.bot {
        background-color: #475063
    }
    .chat-message .avatar {
      width: 20%;
    }
    .chat-message .avatar img {
      max-width: 78px;
      max-height: 78px;
      border-radius: 50%;
      object-fit: cover;
    }
    .chat-message .message {
      width: 80%;
      padding: 0 1.5rem;
      color: #fff;
    }
    .stButton>button {
        background-color: #a8baba;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #a8baba;
    }
    .sidebar .element-container {
        background-color: #a8baba;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .new-chat-button {
        border: none;
        background: #a8baba;
        color: #a8baba;
        cursor: pointer;
        font-size: 1.5em;
    }
    .new-chat-button:hover {
        color: #a8baba;
    }
    .sidebar-title {
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("RAG based AI Chat Model")

    if 'user_id' not in st.session_state:
        st.session_state.user_id = None

    if st.button("New User"):
        st.session_state.user_id = np.random.randint(1000, 9999)
        st.session_state.conversation = []
        st.session_state.messages = []
        st.experimental_rerun()

    if st.session_state.user_id is None:
        st.write("Please click 'New User' to start a new conversation.")
        return

    st.session_state.conversation = load_from_db(st.session_state.user_id)
    
    sidebar_title_col, new_chat_col = st.sidebar.columns([3, 1])
    
    with sidebar_title_col:
        st.sidebar.header("Chat History")
    
    with new_chat_col:
        if st.sidebar.button("➕", key="new_chat", help="Start a new chat"):
            st.session_state.messages = []
            st.session_state.conversation = []
            st.experimental_rerun()

    for i, (user_q, bot_a) in enumerate(st.session_state.conversation):
        with st.sidebar.expander(f"Q: {user_q}", expanded=False):
            st.write(f"A: {bot_a}")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file is not None:
        original_text = read_pdf(uploaded_file)
        st.session_state['original_text'] = original_text
    elif 'original_text' not in st.session_state:
        st.write("Please upload a PDF to start.")

    if 'original_text' in st.session_state:
        chunks = split_text_into_chunks(st.session_state['original_text'])
        vectorizer, embeddings = create_embeddings(chunks)

        if prompt := st.chat_input("What is your question?"):
            # Detect the type of input
            intent = detect_intent(prompt)

            if intent == "greeting":
                bot_response = handle_greeting()
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                st.chat_message("assistant").markdown(bot_response)
            else:
                # Handle question-answering
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    try:
                        relevant_chunks = retrieve_relevant_chunks(prompt, vectorizer, embeddings, chunks)
                        context = "\n".join(relevant_chunks)
                        bot_response = generate_response(prompt, context)
                        
                        for chunk in bot_response.split():
                            full_response += chunk + " "
                            message_placeholder.markdown(full_response + "▌")
                        message_placeholder.markdown(full_response)
                    except Exception as e:
                        full_response = f"An error occurred: {str(e)}"
                        message_placeholder.markdown(full_response)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.session_state.conversation.append((prompt, full_response))
                save_to_db(st.session_state.user_id, prompt, full_response)

if __name__ == '__main__':
    init_db()
    main()
