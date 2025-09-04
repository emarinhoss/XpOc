# Use official Python runtime as base image
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install package
RUN pip install -e .

# Create data directories
RUN mkdir -p data/raw data/processed

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEVICE=cpu

# Default command
CMD ["python", "scripts/run_pipeline.py", "--config", "config/config.yaml"]