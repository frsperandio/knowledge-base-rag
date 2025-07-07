# Use official Python image
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (if exists) and install
COPY requirements.txt ./
RUN pip install --no-cache-dir --default-timeout=300 -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose Gradio default port
EXPOSE 7860

# Run the app
CMD ["python", "main.py"]
