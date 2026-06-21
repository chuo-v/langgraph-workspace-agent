FROM python:3.11-slim

# install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# to bypass "dubious ownership" block for mounted volumes
RUN git config --global --add safe.directory '*'

# set working directory for the application
WORKDIR /app

# install dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy application source code
COPY src/ ./src/

# expose standard FastAPI port
EXPOSE 8000

# start uvicorn server binding to all internal container interfaces
CMD ["uvicorn", "src.workspace_agent.main:app", "--host", "0.0.0.0", "--port", "8000"]