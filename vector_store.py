import os
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

# Where to store embeddings
DB_DIR = "./chroma_pdf_db"

def process_pdf(pdf_path):
    """Load PDF, split into docs, and store in Chroma DB"""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    db = Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    # Use similarity search with top 5 results
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    return retriever
