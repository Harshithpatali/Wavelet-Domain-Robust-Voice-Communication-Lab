FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# The Render Docker build context must be the repository root.
COPY call_service/requirements.txt /app/call_service/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/call_service/requirements.txt

COPY call_service/ /app/call_service/

EXPOSE 8000

CMD ["sh", "-c", "uvicorn call_service.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
