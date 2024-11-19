# Medical Chatbot Project

This project is a medical chatbot built using **LangChain**, **FAISS**, **Hugging Face Embeddings**, and **Llama 2** for NLP-based question answering from PDF documents. The chatbot can answer medical-related queries by searching relevant information from a collection of PDF documents.

## Features

- **PDF Document Loading**: Loads medical documents in PDF format from a specified directory.
- **Text Chunking**: Splits the documents into manageable text chunks for efficient search.
- **Embeddings**: Uses Hugging Face embeddings to convert text into vector representations.
- **FAISS Indexing**: Builds an FAISS index for fast similarity search.
- **Llama 2 Model**: Utilizes Llama 2 for generating context-aware answers to queries.
- **Chat Interface**: Provides an interactive chat interface for users to ask questions.

## Requirements

To set up the project, you need to have a Python environment with the required dependencies. You can create a virtual environment and install the required packages by following these steps:

### Setup Instructions

1. **Clone the Repository**
   
   Clone this repository to your local machine using:
   ```bash
   git clone https://github.com/Sumit10118/medical-chatbot.git

2. **Create a virtual environment( optional)**
   python -m venv mchatbot

3. **Install the requirements**
   pip install -r requirements.txt

4. **Run the app**
   python app.py

### Model Details
- **Embeddings Model**: sentence-transformers/all-MiniLM-L6-v2
- **Llama 2 Model**: llama-2-7b-chat.ggmlv3.q4_0.bin (make sure to provide the correct path to the model file)