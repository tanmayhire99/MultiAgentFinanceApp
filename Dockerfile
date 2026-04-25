# FinAI Multi-Agent Financial Advisor - Production Dockerfile
# Based on Python 3.11 slim for optimal size and performance

FROM python:3.11-slim

# Cache busting to ensure latest code is copied
ARG CACHE_BUST=default
RUN echo "Cache bust: $CACHE_BUST"

# Set metadata
LABEL maintainer="FinAI Team"
LABEL version="1.0.0"
LABEL description="Production-ready multi-agent financial advisory system"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy minimal production requirements
COPY requirements.prod.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash finai && \
    chown -R finai:finai /app

# Switch to non-root user
USER finai

# Expose the port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
