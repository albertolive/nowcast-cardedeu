# nowcast-cardedeu-cron (Cloudflare Worker)

Replaces the ClawCloud container's prediction loop. Every 10 minutes the
Worker fires `workflow_dispatch` on `nowcast.yml`, which runs `predict_now.py`
in GitHub Actions and pushes state back to the repo.

## Why a Worker (and not GitHub's native `schedule:`)

GitHub Actions cron is best-effort. On this repo it produced 30–80 min gaps
instead of 10 min (see commit `16b69cf`). Cloudflare cron triggers are
precise. The Worker free tier covers ~144 invocations/day by 3 orders of
magnitude.

> **2026-04-27 status:** CF cron is silently disabled on this account.
> Verified via `AccountWorkersInvocationsScheduled` GraphQL dataset: 0
> scheduled invocations across the entire account in the last 5 days,
> despite a registered `*/10 * * * *` trigger and a per-minute diagnostic
> deploy that produced 0 firings in 4 min. Account is verified, super-admin,
> not suspended, on the standard usage model. Per docs, Free plan supports
> 5 cron triggers/account with no billing requirement. Root cause is
> CF-internal — needs support ticket.
>
> Stopgap: trigger via [cron-job.org](https://cron-job.org) hitting the
> Worker's `/dispatch` endpoint. Worker stays deployed as the receiver, so
> when CF cron is fixed we can drop cron-job.org with no code change.

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

## Meteocat quota protection

The prediction job still runs every 10 minutes. Meteocat requests are separate
from that cadence: successful responses are cached per endpoint, and a single
HTTP 429 activates a shared 60-minute breaker for XEMA, municipal Predicció and
XDDE. The local JSON cache uses an advisory lock and atomic replacement, and a
bounded stale fallback is used only where the caller has an explicit safe age:
120 minutes for XEMA, 360 for municipal forecast and 180 for lightning.

This protects a runner from hammering the API, but it is not a provider-quota
guarantee. The breaker survives between runs only when the prediction job
successfully commits and pushes the cache; GitHub/network/provider behavior
still requires monitoring after deployment. No offline test can prove the
SMC account's live quota.
