FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
