# Use Python 3.10 slim as the base image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies for psycopg2 and transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies file and install packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Copy application source code files and prompt
COPY *.py /app/
COPY rag_system_prompt.md /app/


# Copy input, cache, and chromadb directories
COPY input/ /app/input/
COPY .cache/ /app/.cache/
COPY chromadb/ /app/chromadb/

# Default command to run the Graph RAG server
CMD ["python", "system_rag_server.py", "serve"]
