FROM python:3.12-slim

WORKDIR /app/

COPY requirements.txt /app/

RUN pip install -r requirements.txt --no-cache-dir

COPY . /app/

EXPOSE 8050
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:8050", "--workers", "1", "--timeout", "120"]