FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .

RUN chmod +x train.py

ENV SAGEMAKER_PROGRAM train.py

ENTRYPOINT ["python", "train.py"]
