FROM python:3.11-slim

# System deps for pdfplumber, lxml, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY backend/ ./backend/
COPY ingestion/ ./ingestion/
COPY scripts/ ./scripts/
COPY frontend/ ./frontend/

# Create data directories
RUN mkdir -p data/raw_docs

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
