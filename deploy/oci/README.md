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

## Step 0 — create the VM

### Path B: GCP e2-micro (always free, never expires)

Hard limits of the free tier: **us-west1 / us-central1 / us-east1 only**
(any other region bills normally), 30 GB **pd-standard** disk, one free
external IP on this instance. 1 GB RAM — `setup.sh` adds 2 GB swap
automatically; the ML stack fits.

```bash
gcloud compute instances create nowcast-vm \
  --machine-type=e2-micro --zone=us-east1-b \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --boot-disk-size=30GB --boot-disk-type=pd-standard
```

US latency to Meteocat/AEMET is ~120 ms per call — irrelevant at a 10-min
cadence. Then continue at Step 1.

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

## Step 2 — `.env`

Copy `.env.template` → `.env`, fill values. They are the SAME values already
configured as the repo's Actions Secrets (Settings → Secrets → Actions).
`chmod 600 .env`.

## Step 3 — Verify

```bash
docker compose logs -f nowcast        # expect: 📡 Data server started … first 📊 Prediction commit within ~10 min
git -C /opt/nowcast log --oneline -3  # bot commits "📊 Prediction <ts>" appearing every 10 min
curl -s localhost/data/latest_prediction.json | head -c 200
```

Then confirm the site updates: the dashboard reads
`raw.githubusercontent.com/albertolive/nowcast-cardedeu/main/data/latest_prediction.json`
— no inbound ports needed on the VM (firewall can stay closed).

## Step 4 — Retire the old path (only AFTER step 3 is green)

1. **CF Worker**: pause its cron trigger (`wrangler triggers` or dashboard)
   — stop paying dispatches.
2. **Watchdog heartbeat**: `watchdog.yml` check #2 counts workflow_dispatch
   runs — those stop now, so it would false-alarm. Apply
   `watchdog-heartbeat.patch.md` (swaps the dispatch-count for a
   last-commit-age check on `data/latest_prediction.json`; checks #1 and #3
   untouched).
3. **nowcast.yml**: delete the `daily_summary` / `accuracy_report` schedules
   (the container does both itself). KEEP the Sunday `retrain` schedule and
   keep `predict` dispatchable manually as an emergency lever.
4. Optional trim while you're at it: `test.yml` in meteo-brief fires on
   every push (~250 min/mo); scope it to PRs + `scripts/**` paths.

## Ops notes

- Reboots: compose `restart: unless-stopped` self-heals; the entrypoint
  resyncs state from the repo on start, so nothing is lost overnight.
- Logs: journald-equivalent via docker json-file, capped in compose.
- **Idle reclaim**: OCI may reclaim Always-Free instances below ~20%
  utilization for 30+ days. The 10-min loop + data server normally counts
  as activity; if you ever see a reclaim notice, add a trivial load cron.
- Model artifacts: retrains commit to `models/` from Actions; the container
  hot-reloads them from origin every cycle — no model sync needed.
