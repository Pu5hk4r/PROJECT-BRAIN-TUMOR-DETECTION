FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /code

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY main.py .
COPY models/ models/
COPY templates/ templates/
COPY static/ static/


RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /code
USER appuser

EXPOSE 7860


CMD ["gunicorn", "main:app", "-b", "0.0.0.0:7860"]