FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    curl build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (cached layer — copy requirements first)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App source
COPY . .

# Create runtime directories
RUN mkdir -p artifacts/bundle logs

# Set PYTHONPATH so `recsys` package is importable
ENV PYTHONPATH=/app/src

# Non-root user for security
RUN useradd -m recsys && chown -R recsys:recsys /app
USER recsys

EXPOSE 8000

# Health check — matches docker-compose healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "recsys.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--timeout-keep-alive", "30"]
