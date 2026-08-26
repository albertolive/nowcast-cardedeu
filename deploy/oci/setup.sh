#!/usr/bin/env bash
# Bootstrap an Oracle Always-Free A1 (Ubuntu 22.04 aarch64) to run the nowcast container.
# Idempotent: safe to re-run. Expects this script's folder to also contain
# .env (from .env.template) and docker-compose.yml.
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="/opt/nowcast"
RUN_DIR="/opt/nowcast-deploy"

[ -f "$DEPLOY_DIR/.env" ] || { echo "missing $DEPLOY_DIR/.env — copy .env.template next to setup.sh and fill it first"; exit 1; }
set -a; source "$DEPLOY_DIR/.env"; set +a
[ -n "${GIT_TOKEN:-}" ] || { echo "GIT_TOKEN empty in .env"; exit 1; }
[ "$(id -u)" = "0" ] || { echo "run as root (sudo)"; exit 1; }

echo "== docker =="
if ! command -v docker >/dev/null; then
  apt-get update -qq && apt-get install -y -qq ca-certificates curl git tzdata rsync
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" > /etc/apt/sources.list.d/docker.list
  apt-get update -qq && apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
fi
systemctl enable --now docker >/dev/null

echo "== swap (required on 1 GB boxes like GCP e2-micro; skipped if present) =="
if ! swapon --show=NAME 2>/dev/null | grep -q .; then
  fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
sysctl -qw vm.swappiness=10
grep -q '^vm.swappiness' /etc/sysctl.conf || echo 'vm.swappiness=10' >> /etc/sysctl.conf

echo "== repo at $REPO_DIR =="
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone --depth=1 "https://x-access-token:${GIT_TOKEN}@github.com/${GIT_REPO}.git" "$REPO_DIR"
else
  git -C "$REPO_DIR" fetch origin main && git -C "$REPO_DIR" reset --hard origin/main
fi

echo "== runtime dir at $RUN_DIR =="
mkdir -p "$RUN_DIR"
cp "$DEPLOY_DIR/docker-compose.yml" "$RUN_DIR/"
cp "$DEPLOY_DIR/.env" "$RUN_DIR/"
chmod 600 "$RUN_DIR/.env"

echo "== build + start =="
docker compose -f "$RUN_DIR/docker-compose.yml" up -d --build

echo "== status =="
sleep 5
docker ps --filter name=nowcast --format '{{.Names}}: {{.Status}}'
docker logs nowcast --tail 15 || true
echo
echo "Done. Follow with:  docker logs -f nowcast"
