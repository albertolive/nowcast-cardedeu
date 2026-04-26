# nowcast-cardedeu-cron (Cloudflare Worker)

Replaces the ClawCloud container's prediction loop. Every 10 minutes the
Worker fires `workflow_dispatch` on `nowcast.yml`, which runs `predict_now.py`
in GitHub Actions and pushes state back to the repo.

## Why a Worker (and not GitHub's native `schedule:`)

GitHub Actions cron is best-effort. On this repo it produced 30–80 min gaps
instead of 10 min (see commit `16b69cf`). Cloudflare cron triggers are
precise. The Worker free tier covers ~144 invocations/day by 3 orders of
magnitude.

## Deploy

Prereqs: a Cloudflare account (free) and `wrangler` CLI (`npm i -g wrangler`).

```bash
cd worker
wrangler login
wrangler secret put GH_TOKEN   # paste a fine-grained PAT, see below
wrangler deploy
```

### GH_TOKEN

Create a **fine-grained** personal access token at
<https://github.com/settings/personal-access-tokens/new>:

- Resource owner: `albertolive`
- Repository access: only `nowcast-cardedeu`
- Repository permissions: **Actions** → **Read and write**
- Expiration: max allowed (1 year). Set a calendar reminder to rotate.

Nothing else. The token only needs to call the `workflow_dispatch` endpoint.

## Smoke test

After deploy, manually trigger one cycle:

```bash
curl -X POST https://nowcast-cardedeu-cron.<your-account>.workers.dev/dispatch
```

A new run should appear within seconds at
<https://github.com/albertolive/nowcast-cardedeu/actions/workflows/nowcast.yml>.

## Observability

- **Cloudflare side:** Workers → `nowcast-cardedeu-cron` → Logs (live tail).
  Failed dispatches throw and show up here.
- **GitHub side:** the `predict` job runs as before. Failures still trigger
  the existing Telegram alert in `nowcast.yml`.
- **Independent backstop:** `.github/workflows/watchdog.yml` runs hourly,
  checks `latest_prediction.json` is < 30 min old, and alerts via Telegram
  if not. Catches silent failure of *this* Worker.
