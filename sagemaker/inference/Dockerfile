FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    MODEL_SERVER_WORKERS=1 \
    MODEL_SERVER_TIMEOUT=60

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    nginx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/program

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY inference.py .
COPY serve .

RUN chmod +x serve

ENV SAGEMAKER_PROGRAM inference.py
ENV PATH="/opt/program:${PATH}"

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/ping || exit 1

ENTRYPOINT ["python", "-m", "sagemaker_inference.serving"]
