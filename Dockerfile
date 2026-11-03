FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir

COPY vertex/utils.py /app/

COPY vertex/train.py /app/

ENV PYTHONPATH=/app