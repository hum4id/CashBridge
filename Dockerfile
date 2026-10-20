# CashBridge — production image (FastAPI + Jinja2). Python 3.12 slim, non-root.
FROM python:3.12-slim AS app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app ./app
COPY cashbridge ./cashbridge
COPY scripts ./scripts
COPY data/external ./data/external
RUN useradd --create-home --uid 10001 cb && mkdir -p /data /srv/data/samples && chown -R cb /data /srv/data
USER cb
ENV CASHBRIDGE_DB=/data/cashbridge.db APP_ENV=production MOCK_LLM=1
EXPOSE 8000
# Seeds the demo data on first start if the database is empty, then serves.
CMD ["sh", "-c", "python scripts/seed_data.py --if-empty && uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000 --proxy-headers"]
