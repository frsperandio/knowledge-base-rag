#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import glob
from dotenv import load_dotenv
import gradio as gr


# In[ ]:


from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredHTMLLoader,
    UnstructuredWordDocumentLoader
)
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


# In[ ]:


import shutil


# In[ ]:


MODEL = "gpt-4o-mini"
db_name = "vector_db"


# In[ ]:


load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")


# In[ ]:


documents = []

file_loaders = {
    ".pdf": UnstructuredPDFLoader,
    ".html": UnstructuredHTMLLoader,
    ".htm": UnstructuredHTMLLoader,
    ".docx": UnstructuredWordDocumentLoader
}

folder_path = "knowledge-base"

# Encontra todos os arquivos com extensões suportadas
file_paths = glob.glob(f"{folder_path}/**/*.*", recursive=True)

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
            print(f"Erro ao carregar {file_path}: {e}")


# In[ ]:


text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)


# In[ ]:


embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

#if os.path.exists(db_name):
#    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
#vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)


# In[ ]:


llm = ChatOpenAI(temperature=0.7, model_name=MODEL)

retriever = vectorstore.as_retriever()

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)


# In[ ]:


def chat(message, history):
    result = conversation_chain.invoke({"question": message})
    return result["answer"]


# In[ ]:


def process_file(file):
    try:
        print(f"file: {file}")
        file_name = os.path.basename(file)
        save_dir = "./knowledge-base"
        os.makedirs(save_dir, exist_ok=True)
        destination = os.path.join(save_dir, file_name)
        print(f"destination: {destination}")
        shutil.copy(file.name, destination)
        gr.update()
        return f"File uploaded and saved to: {destination}"
    except Exception as e:
        return f"Error!: {str(e)}"

chat_interface = gr.ChatInterface(chat, 
                                  type="messages")

with gr.Blocks() as explorer_interface:
    file_explorer = gr.FileExplorer(root_dir="./knowledge-base")
    file_input = gr.File(label="Click to Upload", file_types=[".txt", ".pdf", ".docx"])
    output = gr.Textbox(label="Result")
    upload_button = gr.Button("Submit File")
    upload_button.click(fn=process_file, inputs=file_input, outputs=output)

view = gr.TabbedInterface([chat_interface, explorer_interface], ["Chat", "File Explorer"])

view.launch(inbrowser=True)

