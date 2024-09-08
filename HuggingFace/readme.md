This is an implementation of a Retrieval-Augmented Generation (RAG) chatbot using Hugging Face's transformers library.

Features

* Integration with Hugging Face's transformers library
* DF file processing
* TF-IDF-based text chunking and retrieval
* Streamlit-based user interface
* Chat history storage using SQLite

Prerequisites

* Python 3.7+
* Hugging Face transformers library

Installation

* Clone the repository
* Navigate to the Hugging Face folder

Install the required packages:

* pip install streamlit
* pip install transformers
* pip install torch
* pip install PyMuPDF
* pip install numpy
* pip install scikit-learn
* pip install sqlite3

Usage

* Run the script: streamlit run RAG_hf.py

* Follow the on-screen instructions to interact with the chatbot
  
* Configuration
You may need to adjust the following parameters in the script:

model: The Hugging Face model used for text generation (currently set to "google/flan-t5-large")
