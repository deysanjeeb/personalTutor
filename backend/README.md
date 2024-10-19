# PDF QnA Extraction API

This FastAPI-based service is designed to handle PDF file uploads, extract content from PDF pages, and generate question-answer (QnA) pairs from the content using a machine learning model. It also provides a simple chat interface to interact with the model for Q&A purposes.

---

## Table of Contents
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)

---

## Features
1. **PDF Upload**: Accepts PDF files, reads the content of specific pages, and extracts text for QnA generation.
2. **QnA Generation**: Uses an LLM (LLaMA-3 model from Ollama) to generate question-answer pairs based on the extracted text.
3. **Embeddings and Storage**: Extracted QnA pairs are converted into embeddings and stored in a ChromaDB persistent database.
4. **Chat Functionality**: Allows users to send chat requests and receive responses using the LLaMA-3 model and the previously stored QnA data.
5. **Streamlit Integration**: Though Streamlit is imported, it's currently not in use in the app.

---

## Tech Stack
- **FastAPI**: Backend framework to manage API endpoints.
- **Pypdf**: Used to extract text from uploaded PDF files.
- **Transformers**: Uses the AutoModel and AutoTokenizer to load and interact with machine learning models.
- **Ollama**: API interaction to generate LLaMA model embeddings and chat responses.
- **ChromaDB**: Persistent vector database to store and query document embeddings.
- **Pydantic**: Validation of request bodies.
- **Dotenv**: Load environment variables securely from `.env` files.
- **Uvicorn**: ASGI server to run FastAPI applications.

---

## Installation

### Prerequisites
- **Python 3.8+**
- **pip** (Python package installer)

### Setup

1. Install required libraries:

   ```bash
   pip -r requirements.txt
   ```
2. Create environment variables

    ```bash
    touch .env
   ```

    ```bash
    GROQ_API_KEY=<your_groq_api_key>
    OLLAMA_API_KEY=<your_ollama_api_key>
    ```