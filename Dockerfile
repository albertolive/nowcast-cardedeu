FROM python:3.12-slim

WORKDIR /app

# Install git for state push-back
RUN apt-get update && apt-get install -y --no-install-recommends git tzdata \
    && rm -rf /var/lib/apt/lists/*

ENV TZ=Europe/Madrid

# Install Python dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY config.py .
COPY src/ src/
COPY scripts/ scripts/
COPY models/ models/
COPY data/notification_state.json data/notification_state.json
COPY data/predictions_log.jsonl data/predictions_log.jsonl
COPY data/latest_prediction.json data/latest_prediction.json
COPY data/aemet_cache.json data/aemet_cache.json
COPY data/meteocat_cache.json data/meteocat_cache.json
COPY docs/ docs/

# Entrypoint
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

EXPOSE 80

CMD ["./docker-entrypoint.sh"]
