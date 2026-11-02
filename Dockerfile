# Signals-only gold bot — Render free WEB service.
# Binds $PORT (Render sets it); the scan loop runs in-process alongside
# the dashboard. No MT5, no Windows, no trade execution.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY config.yaml knowledge/ ./

# Runtime state (journal, logs, news cache) lives under /app/data etc.
RUN mkdir -p data logs reports

# Render injects $PORT; the serve command binds 0.0.0.0:$PORT.
CMD ["python", "-m", "app.main", "serve"]
