#!/bin/bash
# =============================================================================
# Nowcast Cardedeu — Predict runner per Oracle Cloud VM
# Executa predict_now.py i sincronitza resultats a GitHub
# =============================================================================
set -euo pipefail

REPO_DIR="__REPO_DIR__"
ENV_FILE="$REPO_DIR/.env"
LOG_TAG="nowcast-predict"

# ── Carregar secrets ──
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "[$LOG_TAG] ERROR: No s'ha trobat $ENV_FILE" >&2
    exit 1
fi

# ── Control horari: només 6:00-23:00 Barcelona ──
CURRENT_HOUR=$(TZ="Europe/Madrid" date +%H)
if [ "$CURRENT_HOUR" -lt 6 ] || [ "$CURRENT_HOUR" -ge 23 ]; then
    echo "[$LOG_TAG] Fora d'horari ($CURRENT_HOUR:xx Barcelona). Saltant."
    exit 0
fi

cd "$REPO_DIR"

# ── Activar venv ──
source .venv/bin/activate

# ── Actualitzar codi (ràpid, sense bloquejar si falla) ──
git pull --rebase origin main 2>/dev/null || true

# ── Executar predicció ──
echo "[$LOG_TAG] Executant predicció..."
python scripts/predict_now.py
PREDICT_EXIT=$?

if [ $PREDICT_EXIT -ne 0 ]; then
    echo "[$LOG_TAG] ERROR: predict_now.py ha sortit amb codi $PREDICT_EXIT" >&2
    # Notificar fallada per Telegram si els tokens estan configurats
    if [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ]; then
        curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
            -d chat_id="${TELEGRAM_CHAT_ID}" \
            -d parse_mode=HTML \
            -d text="🔴 <b>Nowcast FALLADA</b>: predict_now.py ha fallat a Oracle Cloud VM." \
            >/dev/null 2>&1 || true
    fi
    exit $PREDICT_EXIT
fi

# ── Sincronitzar dades a GitHub ──
echo "[$LOG_TAG] Sincronitzant dades a GitHub..."
cp -f data/latest_prediction.json docs/latest_prediction.json 2>/dev/null || true
cp -f data/predictions_log.jsonl docs/predictions_log.jsonl 2>/dev/null || true

git add data/predictions_log.jsonl data/notification_state.json data/latest_prediction.json
git add data/aemet_cache.json 2>/dev/null || true
git add docs/latest_prediction.json docs/predictions_log.jsonl 2>/dev/null || true

if ! git diff --cached --quiet; then
    git commit -m "📊 Prediction $(date -u +%Y-%m-%dT%H:%M) [oracle-vm]"
    git push || (git pull --rebase origin main && git push) || true
fi

echo "[$LOG_TAG] Completat correctament."
