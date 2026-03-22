#!/bin/bash
set -e

# ── Clone repo for pushing state back ──
REPO_DIR="/tmp/repo"
if [ -n "$GIT_TOKEN" ] && [ -n "$GIT_REPO" ]; then
    git config --global user.name "nowcast-bot"
    git config --global user.email "nowcast-bot@users.noreply.github.com"
    git clone --depth=1 "https://x-access-token:${GIT_TOKEN}@github.com/${GIT_REPO}.git" "$REPO_DIR"
fi

echo "🌦️  Nowcast Cardedeu — Container started ($(date))"
echo "   Running predict_now.py every 10 minutes (6:00–23:00 Barcelona)"

while true; do
    HOUR=$(TZ=Europe/Madrid date +%H)
    
    if [ "$HOUR" -ge 6 ] && [ "$HOUR" -lt 23 ]; then
        echo ""
        echo "━━━ $(date) ━━━"
        python scripts/predict_now.py || echo "⚠️  predict_now.py failed (exit $?)"
        
        # Push state files back to GitHub
        if [ -n "$GIT_TOKEN" ] && [ -n "$GIT_REPO" ] && [ -d "$REPO_DIR" ]; then
            (
                cd "$REPO_DIR"
                git pull --rebase origin main 2>/dev/null || true
                
                # Copy updated data files from app into repo clone
                cp -f /app/data/latest_prediction.json data/latest_prediction.json
                cp -f /app/data/predictions_log.jsonl data/predictions_log.jsonl
                cp -f /app/data/notification_state.json data/notification_state.json
                cp -f /app/data/aemet_cache.json data/aemet_cache.json 2>/dev/null || true
                
                # Copy to docs/ for dashboard
                cp -f /app/data/latest_prediction.json docs/latest_prediction.json
                cp -f /app/data/predictions_log.jsonl docs/predictions_log.jsonl
                
                git add data/predictions_log.jsonl data/notification_state.json \
                        data/latest_prediction.json data/aemet_cache.json \
                        docs/latest_prediction.json docs/predictions_log.jsonl 2>/dev/null || true
                git diff --cached --quiet || git commit -m "📊 Prediction $(date -u +%Y-%m-%dT%H:%M)"
                git push origin main 2>/dev/null || echo "⚠️  git push failed"
            ) || echo "⚠️  git sync failed"
        fi
    else
        echo "💤 Fora d'horari (${HOUR}h). Esperant..."
    fi
    
    sleep 600
done
