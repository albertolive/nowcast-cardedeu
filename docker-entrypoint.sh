#!/bin/bash
set -e

# ── Clone repo for pushing state back ──
REPO_DIR="/tmp/repo"
if [ -n "$GIT_TOKEN" ] && [ -n "$GIT_REPO" ]; then
    git config --global user.name "nowcast-bot"
    git config --global user.email "nowcast-bot@users.noreply.github.com"
    git clone --depth=1 "https://x-access-token:${GIT_TOKEN}@github.com/${GIT_REPO}.git" "$REPO_DIR"

    # Sync latest repo data INTO /app so we don't lose predictions on image rebuild
    echo "📥 Syncing repo data into container..."
    cp -f "$REPO_DIR/data/predictions_log.jsonl" /app/data/predictions_log.jsonl 2>/dev/null || true
    cp -f "$REPO_DIR/data/latest_prediction.json" /app/data/latest_prediction.json 2>/dev/null || true
    cp -f "$REPO_DIR/data/notification_state.json" /app/data/notification_state.json 2>/dev/null || true
    cp -f "$REPO_DIR/data/aemet_cache.json" /app/data/aemet_cache.json 2>/dev/null || true
fi

echo "🌦️  Nowcast Cardedeu — Container started ($(date))"
echo "   Running predict_now.py every 10 minutes (6:00–23:00 Barcelona)"

while true; do
    HOUR=$(TZ=Europe/Madrid date +%H)
    
    if [ "$HOUR" -ge 6 ] && [ "$HOUR" -lt 23 ]; then
        echo ""
        echo "━━━ $(date) ━━━"
        python scripts/predict_now.py || echo "⚠️  predict_now.py failed (exit $?)"
        
        # Push state files back to GitHub (with retry for concurrent pushes)
        if [ -n "$GIT_TOKEN" ] && [ -n "$GIT_REPO" ] && [ -d "$REPO_DIR" ]; then
            push_ok=false
            for attempt in 1 2 3; do
                (
                    cd "$REPO_DIR"

                    # Reset any failed rebase state, then pull fresh
                    git rebase --abort 2>/dev/null || true
                    git fetch origin main
                    git reset --hard origin/main

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
                    git push origin main
                ) && push_ok=true && break
                echo "⚠️  git push attempt $attempt failed, retrying in 5s..."
                sleep 5
            done
            [ "$push_ok" = true ] || echo "⚠️  git sync failed after 3 attempts"
        fi
    else
        echo "💤 Fora d'horari (${HOUR}h). Esperant..."
    fi
    
    sleep 600
done
