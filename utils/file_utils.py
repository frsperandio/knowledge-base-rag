
import os
import shutil
import ast
from rag_pipeline.pipeline import build_rag_pipeline
import gradio as gr

# Directory where uploaded files are stored
file_root = "uploads"
# Temporary directory for file explorer (if needed)
file_root_tmp = "tmp"

# Processes selected files in the file explorer (currently just returns the paths)
def process_file(path_or_paths):
    return path_or_paths

# Handles file uploads, saves to uploads folder, and rebuilds the RAG pipeline
def handle_upload(file):
    global conversation_chain
    if file is None:
        return "No file uploaded"
    file_path = os.path.join(file_root, os.path.basename(file.name))
    shutil.copy(file.name, file_path)
    # Rebuild the pipeline to include the new file
    conversation_chain = build_rag_pipeline(folder_path=file_root)
    return f"File uploaded to: {file_path}"

# Deletes selected files and rebuilds the RAG pipeline
def delete_files(paths):
    global conversation_chain
    if isinstance(paths, str):
        try:
            paths = ast.literal_eval(paths)
        except Exception as e:
            return f"Invalid path format: {str(e)}"
    
    messages = []
    for path in paths:
        try:
            os.remove(path)
            messages.append(f"Deleted: {os.path.basename(path)}")
        except Exception as e:
            messages.append(f"Error deleting {os.path.basename(path)}: {str(e)}")
    # Rebuild the pipeline after deletion
    conversation_chain = build_rag_pipeline(folder_path=file_root)
    return "\n".join(messages)

# Helper functions to refresh the file explorer widgets
def update_file_explorer():
    return gr.FileExplorer(root_dir=file_root_tmp)
def update_file_explorer_2():
    return gr.FileExplorer(root_dir=file_root)