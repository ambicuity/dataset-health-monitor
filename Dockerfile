# Dataset Health Monitor - GitHub App
# ====================================
# Docker image for running the GitHub App

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt requirements-app.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt -r requirements-app.txt

# Copy application code
COPY scripts/ ./scripts/
COPY app/ ./app/
COPY datasets/ ./datasets/
COPY state/ ./state/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HOST=0.0.0.0
ENV PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# Run the application
CMD ["python", "-m", "app.app"]
