FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for model persistence and ensure permissions
RUN mkdir -p persist && chmod 777 persist

# Set environment variables
ENV PORT=8000
ENV HOST=0.0.0.0

# Expose the port that the application will run on
EXPOSE $PORT

# Healthcheck to ensure the application is responding
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:$PORT/ || exit 1

# Command to run the application
CMD ["python", "app.py"]
