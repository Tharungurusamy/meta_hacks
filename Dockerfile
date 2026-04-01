# Build from repository root: docker build -t cybersec-openenv .
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0

COPY requirements.txt pyproject.toml openenv.yaml README.md ./
COPY __init__.py models.py client.py app.py inference.py ./
COPY env ./env
COPY server ./server
COPY data ./data

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD sh -c 'uvicorn server.app:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000}'
