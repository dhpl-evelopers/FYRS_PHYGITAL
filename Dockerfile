# -------------------------
# Stage 1: Base environment
# -------------------------
FROM python:3.11-slim

# Prevent Python from writing .pyc files & buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory inside container
WORKDIR /app

# Install system dependencies for psycopg2 and ONNX
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project into container
COPY . .

# ✅ Copy Excel & asset files to the same directory your ML model expects
# (This path matches what MLService or FastAPI uses on Azure)
RUN mkdir -p /home/site/wwwroot/assets/
COPY assets/ /home/site/wwwroot/assets/

# Add /app to Python path so 'api.main' can be imported
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--timeout", "120", "--bind", "0.0.0.0:8000", "api.main:app"]

