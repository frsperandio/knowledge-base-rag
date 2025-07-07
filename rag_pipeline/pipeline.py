
# Standard library imports
import os
import glob
import shutil
# Load environment variables from .env file
from dotenv import load_dotenv
# Gradio import (not used directly here, but may be used elsewhere)
import gradio as gr

# LangChain document loaders for different file types
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredHTMLLoader,
    UnstructuredWordDocumentLoader
)
from langchain_community.document_loaders import DirectoryLoader, TextLoader
# Vector store for embeddings
from langchain_community.vectorstores import FAISS
# Text splitter for chunking documents
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
# OpenAI LLM and embeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# Memory and conversational chain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Model name for the LLM
MODEL = "gpt-4o-mini"

# Load environment variables and get OpenAI API key
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')

# Print status of API key for debugging
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

# Build a RAG (Retrieval-Augmented Generation) pipeline using LangChain
def build_rag_pipeline(folder_path="knowledge-base"):
    documents = []  # List to hold loaded documents

    # Map file extensions to their respective loaders
    file_loaders = {
        ".pdf": UnstructuredPDFLoader,
        ".html": UnstructuredHTMLLoader,
        ".htm": UnstructuredHTMLLoader,
        ".docx": UnstructuredWordDocumentLoader
    }

    # Recursively find all files in the folder_path
    file_paths = glob.glob(f"{folder_path}/**/*.*", recursive=True)

    # Load documents using the appropriate loader
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        loader_cls = file_loaders.get(ext)
        if loader_cls:
            try:
                loader = loader_cls(file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = file_path
                    doc.metadata["doc_type"] = ext
                    documents.append(doc)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    # Split documents into smaller chunks for embedding
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

    # Initialize the LLM, retriever, and memory for conversation
    llm = ChatOpenAI(temperature=0.7, model_name=MODEL)
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    # Build the conversational retrieval chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return conversation_chain