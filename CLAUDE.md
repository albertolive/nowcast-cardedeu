# nowcast-cardedeu

Hyperlocal rain nowcast for Cardedeu (Vallès Oriental). XGBoost trained on 12 years of
the local MeteoCardedeu station, fused with Open-Meteo NWP, radar, lightning and SMC
sentinel stations, plus deterministic physical rules on top of the model output.

It does **not** forecast from scratch. It takes what the global models say and corrects
their error using patterns learned from local history. Keep that framing when changing
anything: the model's job is bias correction, not prediction.

`README.md` has the architecture, the feature list, the data sources and the setup steps.
`LESSONS.md` has the gotchas and is required reading before touching the pipeline or
diagnosing an incident. Neither is repeated here.

## Commands

```bash
python -m pytest tests/ -m "not network" -v    # offline suite, what CI gates on
python -m pytest tests/ -v                     # includes tests that hit live APIs
python scripts/predict_now.py                  # one prediction run
python scripts/train_model.py                  # retrain (needs the built dataset)
```

Virtualenv is `.venv`. No linter or formatter is configured; match the file you are in.

Training from nothing is three steps in order: `download_history.py`,
`build_dataset.py`, `train_model.py`.

## The repo pushes to itself every 10 minutes

A Cloudflare Worker cron (`worker/wrangler.toml`, `*/10 * * * *`) fires the prediction
run, which commits `📊 Prediction <timestamp>` and pushes. `main` goes stale within
minutes.

**Always `git pull --rebase` before pushing.** Merge commits from this have already
polluted the history several times. The workflow itself uses `git rebase --autostash
origin/main` in a retry loop for the same reason.

Bot commits are ~98% of the history, so `git log` needs
`--author="Albert"` to be readable, and "most active repo" metrics from raw commit counts
are meaningless here.

## Two things that will mislead you

**The ML model cannot see radar or lightning.** 85.8% of the gain sits in four Open-Meteo
NWP features. All 31 radar features and all 7 lightning features have *zero* gain, because
the seven years of training data had NaN there. The physical rules in
`_apply_physical_constraints()` (`src/model/predict.py`) exist to cover that gap until the
feedback loop has enough verified data to retrain. Never reason as if the model "sees" a
storm on radar.

**Stale data disguised as fresh is the recurring failure mode.** Every serious incident
has been a variant: a source serves old data as current and the system believes it. If
radar values do not move at all between runs (identical `nearest_echo_km`, identical
`max_dbz` for over 30 minutes), suspect the source, not the weather. Real rain always
fluctuates. `LESSONS.md` has the four incidents and the mitigations already in place.

## Conventions

- Don't add retries to AEMET. Its 429s are hours-long global saturation of shared
  infrastructure, worst exactly during bad weather. More retries do not fix that; the
  short retry in `_http.py` and the stale cache in `aemet_cache.py` are the defence.
- `PREC` from MeteoCardedeu is a daily running total, not per-interval rain. Converting it
  wrongly once invalidated 52 verifications. Meteocat XEMA `KX` *is* per-interval.
- API calls are gated: Meteocat is only queried when the rain gate sees a signal. Keep new
  data sources behind a gate too.
- Docs and `LESSONS.md` are written in Catalan. Match that when editing them; code and
  commit messages are English.
- New gotchas go in `LESSONS.md`, not here. This file stays short.
