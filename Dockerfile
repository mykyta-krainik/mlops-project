FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY pipelines/ ./pipelines/
COPY monitoring/ ./monitoring/
COPY locustfile.py ./locustfile.py
COPY scripts/ ./scripts/

ENV PYTHONPATH=/app
