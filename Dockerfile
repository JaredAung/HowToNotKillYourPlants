FROM python:3.11-slim

# Build with repository root as context so ``backend/`` and ``resources/`` both exist under /app.
WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY resources/ ./resources/

WORKDIR /app/backend
ENV PYTHONPATH=/app:/app/backend

EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]