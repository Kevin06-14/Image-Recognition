FROM python:3.9-slim

WORKDIR /app

# OpenCV needs these runtime libraries inside a slim Linux image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY src ./src
COPY configs ./configs
COPY .env.example ./.env.example

RUN mkdir -p /app/data/db /app/data/web_uploads /app/outputs/web /app/models

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
