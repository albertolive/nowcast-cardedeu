#!/bin/bash
set -e

# ── Git config for pushing state back ──
if [ -n "$GIT_TOKEN" ]; then
    git config --global user.name "nowcast-bot"
    git config --global user.email "nowcast-bot@users.noreply.github.com"
    git init -b main
    git remote add origin "https://x-access-token:${GIT_TOKEN}@github.com/${GIT_REPO}.git" 2>/dev/null || true
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
        if [ -n "$GIT_TOKEN" ] && [ -n "$GIT_REPO" ]; then
            (
                cd /app
                # Fetch latest to avoid conflicts
                git fetch origin main --depth=1 2>/dev/null || true
                git reset --soft origin/main 2>/dev/null || true
                
                # Copy data to docs/ for dashboard
                cp -f data/latest_prediction.json docs/latest_prediction.json 2>/dev/null || true
                cp -f data/predictions_log.jsonl docs/predictions_log.jsonl 2>/dev/null || true
                
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
