# Docker image for the separate Private Live Call signaling service.
# The Streamlit UI continues to run on Streamlit Community Cloud.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser

COPY call_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy only the signaling server, not the Streamlit app, call widget, or research datasets.
COPY call_service/app.py /app/app.py

USER appuser
EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
