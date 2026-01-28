# Multi-stage build for production deployment
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY setup.py pyproject.toml ./

# Install package
RUN pip install --no-cache-dir -e .

# Create directories for data and models
RUN mkdir -p data/raw data/processed models/artifacts logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_ENV=production
ENV LOG_LEVEL=INFO

# Expose port for API (if using)
EXPOSE 8000

# Default command
CMD ["flight-predictor", "--help"]