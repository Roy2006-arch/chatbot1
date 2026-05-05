FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# We must start uvicorn from /app so it imports backend.main
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
