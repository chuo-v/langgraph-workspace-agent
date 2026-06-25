FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# Fix directory ownership for the mounted workspace
RUN git config --global --add safe.directory '*'

# Set working directory for the application
WORKDIR /app

# Install uv for high-speed package resolution
RUN pip install --no-cache-dir uv

# Copy requirements and install via uv
COPY requirements.txt .
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Expose standard FastAPI port
EXPOSE 8000

# Start uvicorn server binding to all internal container interfaces
CMD ["uvicorn", "src.workspace_agent.main:app", "--host", "0.0.0.0", "--port", "8000"]