
# Standard library imports
import os
import gradio as gr

# Import your RAG pipeline builder
from rag_pipeline.pipeline import build_rag_pipeline
from utils.file_utils import handle_upload, delete_files, process_file, update_file_explorer, update_file_explorer_2

# Directory where uploaded files are stored
file_root = "uploads"
# Temporary directory for file explorer (if needed)
file_root_tmp = "tmp"

# Initialize the RAG pipeline with the uploads folder
conversation_chain = build_rag_pipeline(folder_path=file_root)

# Chat function: handles user messages and returns answers from the RAG pipeline
def chat(message, history):
    result = conversation_chain.invoke({"question": message})
    return result["answer"]

# Load chat examples from a text file for the ChatInterface
examples_path = os.path.join(os.path.dirname(__file__), "chat_examples.txt")
with open(examples_path, "r", encoding="utf-8") as f:
    chat_examples = [line.strip() for line in f if line.strip()]

# Build the Gradio UI
with gr.Blocks() as demo:
    with gr.Row():
        # Left column: Chat interface
        with gr.Column(scale=2):
            gr.Markdown("## Chat Interface")
            chat_interface = gr.ChatInterface(chat, type="messages", examples=chat_examples)
        # Right column: File explorer and file management
        with gr.Column(scale=1):
            gr.Markdown("## File Explorer")
            file_explorer = gr.FileExplorer(label="File Explorer", root_dir=file_root, file_count="multiple")
            selected_files = gr.Textbox(label="Selected file paths")
            file_explorer.change(process_file, inputs=file_explorer, outputs=selected_files)
            # Refresh button for file explorer
            refresh_btn = gr.Button("Refresh")
            refresh_btn.click(update_file_explorer, outputs=file_explorer).then(update_file_explorer_2, outputs=file_explorer)
            # Delete selected files
            delete_btn = gr.Button("Delete")
            delete_out = gr.Textbox(label="Deleted files")
            delete_btn.click(delete_files, inputs=selected_files, outputs=delete_out)
            # File upload widget
            file_upload = gr.File(label="File Upload")
            upload_btn = gr.Button("Upload")
            upload_out = gr.Textbox(label="File Upload Status")
            upload_btn.click(handle_upload, inputs=[file_upload], outputs=[upload_out])

# Launch the Gradio app
demo.launch(server_name="0.0.0.0")