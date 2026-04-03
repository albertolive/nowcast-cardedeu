#!/bin/bash
set -euo pipefail

# Vercel Git integration està desconnectada.
# Els deploys es fan via GitHub Actions (deploy_dashboard job).
# Aquest script es manté com a fallback si es reconnecta la integració Git.

if ! git rev-parse --verify HEAD^ >/dev/null 2>&1; then
  echo "No hi ha HEAD^ disponible; fem deploy."
  exit 1
fi

changed_files="$(git diff --name-only HEAD^ HEAD -- docs vercel.json || true)"
relevant_files="$(printf '%s\n' "$changed_files" | grep -Ev '^(docs/latest_prediction\.json|docs/predictions_log\.jsonl)$' || true)"

if [[ -z "$relevant_files" ]]; then
  echo "Sense canvis de frontend rellevants; saltem el deploy."
  exit 0
fi

echo "Canvis de frontend detectats; fem deploy."
printf '%s\n' "$relevant_files"
exit 1