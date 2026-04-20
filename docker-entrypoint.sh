#!/bin/bash
set -e

# ── Telegram failure notification helper ──
notify_failure() {
    local job_name="$1"
    if [ -n "$TELEGRAM_BOT_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
        curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
            -d chat_id="${TELEGRAM_CHAT_ID}" \
            -d parse_mode=HTML \
            -d text="🔴 <b>Nowcast FALLADA</b>: <code>${job_name}</code> ha fallat al contenidor." \
            > /dev/null 2>&1 || true
    fi
}

# ── Dedup marker: prevent re-running a job if already succeeded today ──
MARKER_DIR="/tmp/job_markers"
mkdir -p "$MARKER_DIR"

job_done_today() {
    local marker="$MARKER_DIR/$1_$(TZ=Europe/Madrid date +%Y-%m-%d)"
    [ -f "$marker" ]
}

mark_job_done() {
    local marker="$MARKER_DIR/$1_$(TZ=Europe/Madrid date +%Y-%m-%d)"
    touch "$marker"
    # Clean old markers (keep last 3 days)
    find "$MARKER_DIR" -name "$1_*" -mtime +3 -delete 2>/dev/null || true
}

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
    cp -f "$REPO_DIR/data/meteocat_cache.json" /app/data/meteocat_cache.json 2>/dev/null || true
fi

echo "🌦️  Nowcast Cardedeu — Container started ($(date))"
echo "   Predict every 10 min (24/7) | Daily summary 7:00 | Accuracy report Mon 8:00"

# Start HTTP data server (serves latest_prediction.json + predictions_log.jsonl
# on port 80 for the dashboard to fetch directly)
python scripts/serve_data.py 80 /app/data &
echo "📡 Data server started on port 80"

# Track last prediction time to enforce minimum interval
LAST_PREDICT_EPOCH=0
MIN_INTERVAL_SECS=480  # 8 minutes minimum between predictions

while true; do
    HOUR=$(TZ=Europe/Madrid date +%H)
    MINUTE=$(TZ=Europe/Madrid date +%M)
    DOW=$(TZ=Europe/Madrid date +%u)  # 1=Monday ... 7=Sunday

    echo ""
    echo "━━━ $(date) ━━━"

    # Auto-update code from GitHub (hot-reload without container restart)
    # NOTE: Long-running background processes (e.g. serve_data.py started at
    # boot) are NOT restarted here. Code that changes in those processes only
    # takes effect on the next container reboot — that's deliberate, because
    # restarting serve_data.py in-loop races on port 80 and kills the pod.
    if [ -d "$REPO_DIR" ]; then
        (
            cd "$REPO_DIR"
            timeout 20 git fetch origin main 2>/dev/null && git reset --hard origin/main 2>/dev/null
        ) && {
            # -T: treat dest as a file, not a dir. Without -T, when dest exists
            # (e.g. /app/src) cp nests the source inside as /app/src/src/... and
            # the baked-in /app/src/ never gets updated — hot-reload silently
            # no-ops and the container keeps running the image's original code.
            cp -rfT "$REPO_DIR/src"     /app/src
            cp -rfT "$REPO_DIR/scripts" /app/scripts
            cp -f   "$REPO_DIR/config.py" /app/config.py
            cp -rfT "$REPO_DIR/models"  /app/models
            cp -rfT "$REPO_DIR/docs"    /app/docs
            # Clear any stale bytecode so subprocess imports pick up the new .py
            find /app/src /app/scripts -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
            echo "🔄 Code updated from GitHub"
        } || echo "⚠️  Code update failed (continuing with current version)"
    fi

    # ── Daily summary (7:00–7:09 Barcelona, once per day) ──
    if [ "$HOUR" -eq 7 ] && [ "${MINUTE#0}" -lt 10 ] && ! job_done_today "daily_summary"; then
        echo "📋 Running daily_summary.py..."
        if python scripts/daily_summary.py; then
            mark_job_done "daily_summary"
            echo "✅ daily_summary completed"
        else
            echo "⚠️  daily_summary.py failed (exit $?)"
            notify_failure "daily_summary"
        fi
    fi

    # ── Accuracy report (Monday 8:00–8:09 Barcelona, once per day) ──
    if [ "$DOW" -eq 1 ] && [ "$HOUR" -eq 8 ] && [ "${MINUTE#0}" -lt 10 ] && ! job_done_today "accuracy_report"; then
        echo "📊 Running accuracy_report.py..."
        if python scripts/accuracy_report.py; then
            mark_job_done "accuracy_report"
            echo "✅ accuracy_report completed"
        else
            echo "⚠️  accuracy_report.py failed (exit $?)"
            notify_failure "accuracy_report"
        fi
    fi

    # ── Prediction (every 10 min, 24/7) ──
    # Enforce minimum interval to prevent rapid-fire predictions
    # (can happen if git push exceeds 10-min window on low CPU)
    now_epoch=$(date +%s)
    elapsed=$(( now_epoch - LAST_PREDICT_EPOCH ))
    if [ "$elapsed" -lt "$MIN_INTERVAL_SECS" ]; then
        remaining=$(( MIN_INTERVAL_SECS - elapsed ))
        echo "⏳ Massa aviat (${elapsed}s des de l'última predicció, mínim ${MIN_INTERVAL_SECS}s). Esperant ${remaining}s..."
        sleep "$remaining"
    fi
    LAST_PREDICT_EPOCH=$(date +%s)
    python scripts/predict_now.py || {
        echo "⚠️  predict_now.py failed (exit $?)"
        notify_failure "predict_now"
    }

    # Push state files back to GitHub (with retry for concurrent pushes)
    # Small files every cycle (~60KB); JSONL only hourly (~13MB) to minimize git bloat.
    # Dashboard gets real-time data from the HTTP server on port 80.
    if [ -n "$GIT_TOKEN" ] && [ -n "$GIT_REPO" ] && [ -d "$REPO_DIR" ]; then
        PUSH_MINUTE=$(TZ=Europe/Madrid date +%M)
        push_ok=false
        for attempt in 1 2 3; do
            (
                cd "$REPO_DIR"

                # Reset any failed rebase state, then pull fresh
                git rebase --abort 2>/dev/null || true
                timeout 30 git fetch origin main
                git reset --hard origin/main

                # Always push small files (latest prediction + state + caches)
                cp -f /app/data/latest_prediction.json data/latest_prediction.json
                cp -f /app/data/latest_prediction.json docs/latest_prediction.json
                cp -f /app/data/notification_state.json data/notification_state.json
                cp -f /app/data/aemet_cache.json data/aemet_cache.json 2>/dev/null || true
                cp -f /app/data/meteocat_cache.json data/meteocat_cache.json 2>/dev/null || true

                git add data/latest_prediction.json docs/latest_prediction.json \
                        data/notification_state.json data/aemet_cache.json \
                        data/meteocat_cache.json 2>/dev/null || true
                # Push slim JSONL to docs/ every cycle (~30KB) so GitHub Pages
                # serves fresh history when the container is unreachable.
                python3 -c "
import json
KEEP = {'timestamp','probability_pct','rain_category','verified','actual_rain','actual_rain_mm','correct'}
with open('/app/data/predictions_log.jsonl') as f:
    for raw in f:
        raw = raw.strip()
        if not raw: continue
        try:
            obj = json.loads(raw)
            print(json.dumps({k: obj[k] for k in KEEP if k in obj}, separators=(',',':')))
        except Exception:
            pass
" > docs/predictions_log.jsonl
                git add docs/predictions_log.jsonl
                # Push JSONL once daily at 3:00 (before retrain cron at 3:00 UTC Sunday).
                # Dashboard reads JSONL from the HTTP server, not git.
                # On container restart, startup sync recovers the last pushed version.
                PUSH_HOUR=$(TZ=Europe/Madrid date +%H)
                if [ "$PUSH_HOUR" -eq 3 ] && [ "${PUSH_MINUTE#0}" -lt 10 ]; then
                    # Truncate JSONL to ~35 days before pushing
                    MAX_JSONL_LINES=5000
                    JSONL_FILE=/app/data/predictions_log.jsonl
                    line_count=$(wc -l < "$JSONL_FILE" 2>/dev/null || echo 0)
                    if [ "$line_count" -gt "$MAX_JSONL_LINES" ]; then
                        echo "✂️  Truncating JSONL: ${line_count} → ${MAX_JSONL_LINES} lines"
                        tail -n "$MAX_JSONL_LINES" "$JSONL_FILE" > "${JSONL_FILE}.tmp" && mv "${JSONL_FILE}.tmp" "$JSONL_FILE"
                    fi
                    # Push slim JSONL (dashboard fields only) so the git fallback
                    # is ~300 KB instead of the full ~15 MB with 211 feature columns.
                    python3 -c "
import json, sys
KEEP = {'timestamp','probability_pct','rain_category','verified','actual_rain','actual_rain_mm','correct'}
with open('/app/data/predictions_log.jsonl') as f:
    for raw in f:
        raw = raw.strip()
        if not raw: continue
        try:
            obj = json.loads(raw)
            print(json.dumps({k: obj[k] for k in KEEP if k in obj}, separators=(',',':')))
        except Exception:
            pass
" > data/predictions_log.jsonl
                    git add data/predictions_log.jsonl
                fi

                git diff --cached --quiet || git commit -m "📊 Prediction $(date -u +%Y-%m-%dT%H:%M)"
                timeout 30 git push origin main
            ) && push_ok=true && break
            echo "⚠️  git push attempt $attempt failed, retrying in 5s..."
            sleep 5
        done
        [ "$push_ok" = true ] || echo "⚠️  git sync failed after 3 attempts"

        # Compact git objects to prevent .git growth from repeated fetch/reset cycles
        # (144 fetches/day on a shallow clone can balloon .git/objects/pack/)
        if [ "$push_ok" = true ]; then
            (cd "$REPO_DIR" && git gc --prune=now 2>/dev/null) || true
        fi
    fi
    
    # Sleep until next 10-minute mark (e.g. :00, :10, :20, :30, :40, :50)
    now=$(date +%s)
    next=$(( (now / 600 + 1) * 600 ))
    sleep_secs=$(( next - $(date +%s) ))
    # Minimum sleep of 480s prevents rapid-fire if cycle exceeds 10 min
    [ "$sleep_secs" -lt 120 ] && sleep_secs=$(( next + 600 - $(date +%s) ))
    [ "$sleep_secs" -le 0 ] && sleep_secs=480
    echo "⏱️  Next run in ${sleep_secs}s (at $(TZ=Europe/Madrid date -d @$next +%H:%M 2>/dev/null || date -r $next +%H:%M))"
    sleep "$sleep_secs"
done
