# Nowcast Cardedeu — free 24/7 runtime migration

Move the 24/7 prediction runtime off GitHub-hosted runners (~15,300 billed
min/month — the whole account's quota). Two €0 paths:

- **Path A (zero migration): make this repo public.** Actions on public
  repos is free and unlimited — the CF Worker, watchdog and every workflow
  keep working unchanged, today. Revert with one command. Cost: the model,
  features and training data become world-readable. Secrets are safe
  either way (they live in GH Secrets / CF env; verified: no tokens in
  tracked files).
  ```bash
  gh repo edit albertolive/nowcast-cardedeu --visibility public \
    --accept-visibility-change-consequences
  ```
- **Path B (private code): Google Cloud always-free e2-micro VM.** The OCI
  equivalent that actually exists in 2026 after Oracle halved/enforced its
  tier. Steps below.

Either way the account lands back under 2,000 private-repo min/month and
meteo-brief's `daily-brief.yml` returns as production Sep 1.

## Why this shape

| Before | After |
|---|---|
| CF Worker dispatches workflow_dispatch every 10 min | VM runs the same container continuously |
| Each tick rents a ubuntu runner: checkout + pip install + ~10 s of Python | Zero rented compute; ticks are function calls in-process |
| Watchdog on Actions cron (hourly) | Unchanged — deliberate cross-platform backstop |
| Sunday retrain on Actions cron | Unchanged (not in the container loop; a few min/month) |

After migration, account Actions usage drops to ~1,400 min/month — inside
the 2,000 free tier again, so meteo-brief's `daily-brief.yml` can return as
production on Sep 1.

## Step 0 — create the VM (actual: esdeveniments:us-east1-b 34.139.5.189)

### Path B: GCP e2-micro (always free, never expires) - implementat 2026-08-30

Hard limits of the free tier: **us-west1 / us-central1 / us-east1 only**
(any other region bills normally, `europe-southwest1-a` Madrid `~$8/mo`), 30 GB **pd-standard** disk, one free external IP. 1 GB RAM — `setup.sh` adds 2 GB swap.

Actual: `esdeveniments` `381787440315` `billingEnabled:true` (ja tenia `compute.googleapis.com` per BigQuery/CloudRun, `Instances 0/24`). Nou projecte `nowcast-cardedeu-20260830` va fallar `billing quota exceeded 5/5` `016A84-EE8812-1C8C5C` - esborrat. Reutilitzat `esdeveniments` per `0€` marginal (separat de `que-fer` Hetzner Coolify).

```bash
gcloud compute instances create nowcast-vm --project=esdeveniments --zone=us-east1-b \
  --machine-type=e2-micro --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --boot-disk-size=30GB --boot-disk-type=pd-standard --tags=nowcast
```

US latency `~120ms` vs `europe-southwest1` `~20ms` irrelevant a 10-min. `esdeveniments-3` `billingEnabled:false` no serveix. Llavors pas 1.

### (Original) Oracle Always Free A1

Only if signup ever works for you — card verification rejects many cards.
Shape **VM.Standard.A1.Flex, exactly 2 OCPU / 12 GB** (the Always-Free cap;
larger is auto-terminated since Aug 2026), Ubuntu 22.04 aarch64, home
region permanent (pick eu-madrid), retry A1 capacity off-peak.

## Step 1 — Bootstrap the VM

SSH in, drop this folder + a `.env` next to it, then:

```bash
sudo bash setup.sh          # installs docker, clones repo, builds image
```

`setup.sh` reads `GIT_TOKEN` from `.env` for the private clone. Everything
is idempotent — rerunning is safe.

## Step 2 — `.env` (guardat local 2026-08-30)

Copia `.env` ja creat a arrel (`nowcast-cardedeu/.env` `chmod 600` gitignored) i `deploy/oci/.env` - 10 vars: `GIT_TOKEN` (`gho_...` `repo` push), `GIT_REPO`, `TELEGRAM_BOT_TOKEN=8631860454:AAH...` `CHAT_ID=-1003766942798` (channel `MeteoBot Cardedeu`), `METEOCAT fTVz...`, `AEMET eyJ...`, `GATEWAY_TOKEN=a3a2...` (=`ai-gateway/.env` `AI_GATEWAY_API_KEY`), `OPENROUTER/GEMINI/GROQ`. Sense `TELEGRAM_CHAT_ID` el VM corre però no alerta. `METEOCAT/AEMET` opcionals (NWP fallback). Sense `.env` local, `gh auth token` reutilitzat temporalment.

## Step 3 — Verify

```bash
docker compose logs -f nowcast        # expect: 📡 Data server started … first 📊 Prediction commit within ~10 min
git -C /opt/nowcast log --oneline -3  # bot commits "📊 Prediction <ts>" appearing every 10 min
curl -s localhost/data/latest_prediction.json | head -c 200
```

Then confirm the site updates: the dashboard reads
`raw.githubusercontent.com/albertolive/nowcast-cardedeu/main/data/latest_prediction.json`
— no inbound ports needed on the VM (firewall can stay closed).

## Step 4 — Retirar l'antic (fet 2026-08-30)

1. **CF Worker**: no cal - ja mort, `watchdog` ja comptava 0 dispatches.
2. **Watchdog heartbeat**: aplicat `watchdog.yml:90` `Check prediction-commit heartbeat` (`gh api commits?path=data/latest_prediction.json&since=80m` `COUNT>=6` per `watchdog-heartbeat.patch.md`). Alerts actualitzats a `GCP nowcast-vm`.
3. **nowcast.yml:75** `predict` `if:false` (VM 10min free, Actions `16k->800` `PASS` `quota_guard`). `daily_summary/accuracy_report/retrain` queden a Actions (container no els fa).
4. `quota-guard.yml` diari `0 6 * * *` `80% 2400/3000` via `billing/usage`.
5. Vercel `vercel.json:4` `docs` + `docs/app.js:17` `RAW_BASE` fallback 30min + `setInterval 5min` = sense redeploy, VM pushes basten.

## Ops notes

- Reboots: compose `restart: unless-stopped` self-heals; the entrypoint
  resyncs state from the repo on start, so nothing is lost overnight.
- Logs: journald-equivalent via docker json-file, capped in compose.
- **Idle reclaim**: OCI may reclaim Always-Free instances below ~20%
  utilization for 30+ days. The 10-min loop + data server normally counts
  as activity; if you ever see a reclaim notice, add a trivial load cron.
- Model artifacts: retrains commit to `models/` from Actions; the container
  hot-reloads them from origin every cycle — no model sync needed.
