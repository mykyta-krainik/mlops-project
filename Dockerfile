FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    locales-all \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY pipelines/ ./pipelines/
COPY monitoring/ ./monitoring/
COPY locustfile.py ./locustfile.py
COPY scripts/ ./scripts/

ENV PYTHONPATH=/app

# SageMaker runs the container as: docker run <image> serve
# Create a `serve` executable in PATH that starts gunicorn
RUN printf '#!/bin/bash\nexec gunicorn --bind 0.0.0.0:8080 --workers 1 --timeout 120 src.serve_sagemaker:app\n' \
    > /usr/local/bin/serve && chmod +x /usr/local/bin/serve

EXPOSE 8080
CMD ["serve"]
