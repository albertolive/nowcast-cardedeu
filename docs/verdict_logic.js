// Verdict text logic. Decides the headline label shown next to the probability
// ring, distinguishing five user-relevant states:
//   - persistence (it's raining now and the model expects it to continue)
//   - rain ending (raining now but model leans dry — rare, the 0.80 floor in
//     predict.py blocks most cases)
//   - uncertain while raining (raining now, probability in the 30-65% band)
//   - anticipation (not raining now but probability ≥65% — the valuable case)
//   - dry / uncertain / probable fallbacks when no trustworthy local signal
//
// The station signal is trusted only when:
//   1. the JSON exposes station_raining_now (post-refactor predictions),
//   2. station_available is not explicitly false (station was online),
//   3. the prediction itself is fresh (≤15 min old — the cron runs every 10).

// Keep in lockstep with the cron interval. 15 min covers normal jitter
// without claiming "plou ara" hours after an outage.
export const STATION_FRESHNESS_MS = 15 * 60 * 1000;

export function isStationStateKnown(d, now = Date.now()) {
  if (d == null || d.station_raining_now == null) return false;
  if (d.station_available === false) return false;
  const ts = d.timestamp ? new Date(d.timestamp).getTime() : NaN;
  if (!Number.isFinite(ts)) return false;
  return now - ts <= STATION_FRESHNESS_MS;
}

export function verdictText(d, now = Date.now()) {
  const cat = d?.rain_category;
  const pct = d?.probability_pct;
  const isProbable = cat === 'probable' || pct >= 65;
  const isDry = cat === 'sec' || pct < 30;

  if (isStationStateKnown(d, now)) {
    if (d.station_raining_now === true) {
      if (isProbable) return '🌧️ Plou ara · continua';
      if (isDry) return '🌧️ Plou ara · acabarà aviat';
      return '🌧️ Plou ara · pot parar aviat';
    }
    if (isProbable) return '⚠️ Pluja imminent';
    if (isDry) return '☀️ No plourà';
    return `🌤️ ${pct}% probabilitat de pluja`;
  }

  if (isProbable) return '🌧️ Pluja probable';
  if (isDry) return '☀️ No plourà';
  return `🌤️ ${pct}% probabilitat de pluja`;
}
