#!/usr/bin/env bash
# Auto-update del nowcast al VM: pull + rebuild NOMÉS quan canvia codi.
# Seguretat: gate d'IC verda, backup de la imatge anterior, health check i rollback.
# Instal·lat com a cron (cada 15 min) per setup.sh. Idempotent i amb lock.
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

REPO_DIR="/opt/nowcast"
RUN_DIR="/opt/nowcast-deploy"
LOG_FILE="/var/log/nowcast-selfupdate.log"
# Rutes que exigeixen rebuild d'imatge. Commits només de dades (push del bot)
# fan fast-forward del repo sense rebuild — el contenidor gestiona el seu estat.
CODE_PATHS='^(src/|scripts/|models/|Dockerfile$|docker-entrypoint\.sh$|requirements\.txt$|config\.py$)'
REPO="albertolive/nowcast-cardedeu"

log() { echo "[self-update $(date -u '+%F %T')] $*"; }

# Rota el log si creix massa
[ -f "$LOG_FILE" ] && [ "$(stat -c%s "$LOG_FILE" 2>/dev/null || echo 0)" -gt 5242880 ] && : > "$LOG_FILE"

# Lock: mai dos self-updates concurrents
exec 9>/run/nowcast-selfupdate.lock
flock -n 9 || exit 0

cd "$REPO_DIR"
git fetch origin main --quiet

LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)
[ "$LOCAL" = "$REMOTE" ] && exit 0

CHANGED=$(git diff --name-only "$LOCAL" "$REMOTE")
log "canvis detectats: $(echo "$CHANGED" | wc -l) fitxers"

if ! echo "$CHANGED" | grep -qE "$CODE_PATHS"; then
    git reset --hard origin/main --quiet
    log "només dades: fast-forward a ${REMOTE:0:8} sense rebuild"
    exit 0
fi

# ── Gate d'IC: no desplegar codi que no hagi passat els tests ──
CI_STATUS=$(curl -fsSL --max-time 20 \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/${REPO}/commits/${REMOTE}/check-runs?per_page=100" \
    | python3 -c "
import json, sys
runs = json.load(sys.stdin).get('check_runs', [])
test = [r for r in runs if r.get('name') == 'test']
if not test:
    print('pending')
elif any(r['status'] != 'completed' for r in test):
    print('pending')
elif any(r.get('conclusion') not in ('success', 'neutral', 'skipped') for r in test):
    print('failure')
else:
    print('success')
" 2>/dev/null || echo "pending")

if [ "$CI_STATUS" != "success" ]; then
    log "IC en estat '$CI_STATUS' per ${REMOTE:0:8} — esperant el proper cicle"
    exit 0
fi
log "IC verda per ${REMOTE:0:8} — reconstruïnt"

git reset --hard origin/main --quiet

# Backup de la imatge en marxa per rollback
CURRENT_IMAGE=$(docker inspect --format '{{.Image}}' nowcast 2>/dev/null || true)
RESTARTS_BEFORE=$(docker inspect --format '{{.RestartCount}}' nowcast 2>/dev/null || echo 0)
if [ -n "$CURRENT_IMAGE" ]; then
    docker tag "$CURRENT_IMAGE" nowcast:backup 2>/dev/null || true
fi

docker compose -f "$RUN_DIR/docker-compose.yml" up -d --build

# Health check: el contenidor ha d'estar corrent 90s sense restart-loop.
# Un ImportError del entrypoint fa sortir el contenidor → restart policy → count.
sleep 90
RUNNING=$(docker inspect --format '{{.State.Running}}' nowcast 2>/dev/null || echo false)
RESTARTS_AFTER=$(docker inspect --format '{{.RestartCount}}' nowcast 2>/dev/null || echo 999)

if [ "$RUNNING" != "true" ] || [ "$RESTARTS_AFTER" -gt "$RESTARTS_BEFORE" ]; then
    log "SALUT FALLIDA (running=$RUNNING, restarts=$RESTARTS_BEFORE->$RESTARTS_AFTER) — FENT ROLLBACK"
    IMG_REF=$(docker inspect --format '{{.Config.Image}}' nowcast 2>/dev/null || true)
    if [ -n "$CURRENT_IMAGE" ] && [ -n "$IMG_REF" ]; then
        docker tag "$CURRENT_IMAGE" "$IMG_REF"
        docker compose -f "$RUN_DIR/docker-compose.yml" up -d --no-build
        log "rollback completat a la imatge anterior"
    else
        log "no es pot fer rollback (imatge anterior desconeguda)"
    fi
    exit 1
fi

log "desplegat ${REMOTE:0:8} correctament (health check OK)"