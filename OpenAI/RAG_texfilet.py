import os
import sqlite3
import streamlit as st
import httpx
from openai import AzureOpenAI
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the Azure OpenAI client
httpx_client = httpx.Client(verify=False)
openai_client = AzureOpenAI(
    api_key="your_api_key",
    api_version="your_api_version",
    azure_endpoint="https://newopenairnd.openai.azure.com/",
    http_client=httpx_client
)

# Paths to text file and database
TEXT_FILE_PATH = os.path.join(os.getcwd(), 'data', 'context.txt')
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

# Function to read a text file
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

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

# Function to generate a response using Azure OpenAI
def generate_response(query, context):
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    response = openai_client.chat.completions.create(
        model="chatgp35test",
        messages=[
            {"role": "system", "content": "You are a very helpful AI assistant that answers questions with a human touch to it ."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    
    return response.choices[0].message.content

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

    if prompt := st.chat_input("What is your question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Read the text file and process
        original_text = read_text_file(TEXT_FILE_PATH)
        chunks = split_text_into_chunks(original_text)
        vectorizer, embeddings = create_embeddings(chunks)

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
