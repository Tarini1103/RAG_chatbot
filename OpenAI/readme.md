OpenAI RAG Chatbot
 Implementations of Retrieval-Augmented Generation (RAG) chatbots using OpenAI's models via the Azure OpenAI API.
 
Files

* RAG_PDF.py: RAG chatbot that processes PDF files as input
* RAG_text.py: RAG chatbot that processes text files as input

Features

* Integration with Azure OpenAI API
* PDF and text file processing
* TF-IDF based text chunking and retrieval
* Streamlit-based user interface
* Chat history storage using SQLite

Prerequisites

* Python 3.7+
* Azure OpenAI API key and endpoint

Installation

Clone the repository
Navigate to the OpenAI folder
Install the required packages:

* pip install streamlit 
* pip install openai
* pip install httpx
* pip installPyMuPDF
* pip install numpy
* pip install scikit-learn
* pip install sqlite3
  
Usage

Set up your Azure OpenAI API credentials in the scripts
Run the desired script:

streamlit run RAG_PDF.py
or
streamlit run RAG_text.py

Follow the on-screen instructions to interact with the chatbot

Configuration
You may need to adjust the following parameters in the scripts:

* api_key: Your Azure OpenAI API key
* api_version: The API version you're using
* azure_endpoint: Your Azure OpenAI endpoint URL
