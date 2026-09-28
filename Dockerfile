# Dockerfile for Hugging Face Spaces / Vercel alternative
# Keep the optional root image aligned with backend CI and constraints.

FROM python:3.12-slim

# Set working directory to /app
WORKDIR /app

# Install system dependencies if required (e.g., for building some python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements
COPY backend/requirements.txt .
COPY backend/constraints.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Copy the backend code into the container
COPY backend/ .

# Expose port 7860 (Default for Hugging Face Spaces)
EXPOSE 7860

# Command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
