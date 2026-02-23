# Dockerfile for Hugging Face Spaces / Vercel alternative
# Uses Python 3.10 slim image

FROM python:3.10-slim

# Set working directory to /app
WORKDIR /app

# Install system dependencies if required (e.g., for building some python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements
COPY backend/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code into the container
COPY backend/ .

# Expose port 7860 (Default for Hugging Face Spaces)
EXPOSE 7860

# Command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
