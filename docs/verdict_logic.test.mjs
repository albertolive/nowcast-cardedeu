import test from "node:test";
import assert from "node:assert/strict";
import { verdictText, isStationStateKnown, STATION_FRESHNESS_MS } from "./verdict_logic.js";

const FRESH_NOW = Date.parse("2026-05-09T16:25:00Z");
const FRESH_TS = "2026-05-09T16:20:00Z"; // 5 min before "now"

function pred(overrides = {}) {
  return {
    timestamp: FRESH_TS,
    probability_pct: 50,
    rain_category: "incert",
    station_available: true,
    station_raining_now: false,
    ...overrides,
  };
}

// ── isStationStateKnown ─────────────────────────────────────────────────

test("station state known: fresh, available, with field", () => {
  assert.equal(isStationStateKnown(pred(), FRESH_NOW), true);
});

test("station state unknown when station_raining_now field missing (legacy JSON)", () => {
  const d = pred();
  delete d.station_raining_now;
  assert.equal(isStationStateKnown(d, FRESH_NOW), false);
});

test("station state unknown when station_raining_now is null", () => {
  assert.equal(isStationStateKnown(pred({ station_raining_now: null }), FRESH_NOW), false);
});

test("station state unknown when station_available is false", () => {
  assert.equal(
    isStationStateKnown(pred({ station_available: false, station_raining_now: true }), FRESH_NOW),
    false
  );
});

test("station_available undefined still counts as available (older JSON without field)", () => {
  const d = pred();
  delete d.station_available;
  assert.equal(isStationStateKnown(d, FRESH_NOW), true);
});

test("station state unknown when prediction is older than freshness window", () => {
  const stale = new Date(FRESH_NOW - STATION_FRESHNESS_MS - 60_000).toISOString();
  assert.equal(isStationStateKnown(pred({ timestamp: stale }), FRESH_NOW), false);
});

test("station state known exactly at freshness boundary", () => {
  const boundary = new Date(FRESH_NOW - STATION_FRESHNESS_MS).toISOString();
  assert.equal(isStationStateKnown(pred({ timestamp: boundary }), FRESH_NOW), true);
});

test("station state unknown when timestamp missing", () => {
  const d = pred();
  delete d.timestamp;
  assert.equal(isStationStateKnown(d, FRESH_NOW), false);
});

test("station state unknown when timestamp unparseable", () => {
  assert.equal(isStationStateKnown(pred({ timestamp: "not-a-date" }), FRESH_NOW), false);
});

test("station state unknown when prediction is null", () => {
  assert.equal(isStationStateKnown(null, FRESH_NOW), false);
});

// ── verdictText: known station, raining now ─────────────────────────────

test("raining now + high probability → persistence", () => {
  const d = pred({ station_raining_now: true, probability_pct: 80, rain_category: "probable" });
  assert.equal(verdictText(d, FRESH_NOW), "🌧️ Plou ara · continua");
});

test("raining now + uncertain → may stop soon", () => {
  const d = pred({ station_raining_now: true, probability_pct: 50, rain_category: "incert" });
  assert.equal(verdictText(d, FRESH_NOW), "🌧️ Plou ara · pot parar aviat");
});

test("raining now + low probability → ending soon (rare due to 0.80 floor)", () => {
  const d = pred({ station_raining_now: true, probability_pct: 10, rain_category: "sec" });
  assert.equal(verdictText(d, FRESH_NOW), "🌧️ Plou ara · acabarà aviat");
});

// ── verdictText: known station, not raining now ─────────────────────────

test("not raining + high probability → anticipation (the valuable case)", () => {
  const d = pred({ station_raining_now: false, probability_pct: 80, rain_category: "probable" });
  assert.equal(verdictText(d, FRESH_NOW), "⚠️ Pluja imminent");
});

test("not raining + uncertain → show probability honestly", () => {
  const d = pred({ station_raining_now: false, probability_pct: 45, rain_category: "incert" });
  assert.equal(verdictText(d, FRESH_NOW), "🌤️ 45% probabilitat de pluja");
});

test("not raining + low probability → dry", () => {
  const d = pred({ station_raining_now: false, probability_pct: 5, rain_category: "sec" });
  assert.equal(verdictText(d, FRESH_NOW), "☀️ No plourà");
});

// ── verdictText: fallback when station state unknown ────────────────────

test("legacy JSON (no station_raining_now) + high prob → generic 'pluja probable'", () => {
  const d = pred({ probability_pct: 80, rain_category: "probable" });
  delete d.station_raining_now;
  assert.equal(verdictText(d, FRESH_NOW), "🌧️ Pluja probable");
});

test("station offline + high prob → falls back to generic phrasing", () => {
  const d = pred({
    station_available: false,
    station_raining_now: true,
    probability_pct: 80,
    rain_category: "probable",
  });
  assert.equal(verdictText(d, FRESH_NOW), "🌧️ Pluja probable");
});

test("stale prediction does not claim 'plou ara' even if field says so", () => {
  const stale = new Date(FRESH_NOW - 60 * 60_000).toISOString();
  const d = pred({
    timestamp: stale,
    station_raining_now: true,
    probability_pct: 80,
    rain_category: "probable",
  });
  assert.equal(verdictText(d, FRESH_NOW), "🌧️ Pluja probable");
});

// ── boundary conditions on probability ──────────────────────────────────

test("probability exactly 65 → probable (raining now branch)", () => {
  const d = pred({ station_raining_now: true, probability_pct: 65, rain_category: "incert" });
  // pct >= 65 takes the "probable" branch
  assert.equal(verdictText(d, FRESH_NOW), "🌧️ Plou ara · continua");
});

test("probability exactly 30 → uncertain", () => {
  const d = pred({ station_raining_now: false, probability_pct: 30, rain_category: "incert" });
  assert.equal(verdictText(d, FRESH_NOW), "🌤️ 30% probabilitat de pluja");
});

test("probability 29.9 → dry", () => {
  const d = pred({ station_raining_now: false, probability_pct: 29.9, rain_category: "sec" });
  assert.equal(verdictText(d, FRESH_NOW), "☀️ No plourà");
});

test("rain_category overrides probability when category is 'probable'", () => {
  // pct alone would be uncertain (50), but cat says probable
  const d = pred({ station_raining_now: false, probability_pct: 50, rain_category: "probable" });
  assert.equal(verdictText(d, FRESH_NOW), "⚠️ Pluja imminent");
});

test("rain_category overrides probability when category is 'sec'", () => {
  const d = pred({ station_raining_now: false, probability_pct: 50, rain_category: "sec" });
  assert.equal(verdictText(d, FRESH_NOW), "☀️ No plourà");
});

// ── missing probability_pct ─────────────────────────────────────────────

test("missing probability_pct in fallback branch → honest 'sense dades' text", () => {
  // Legacy/corrupt entry: no station signal AND no probability.
  const d = { timestamp: FRESH_TS, rain_category: "incert" };
  assert.equal(verdictText(d, FRESH_NOW), "🌫️ Sense dades suficients");
});

test("missing probability_pct with known station signal also yields 'sense dades'", () => {
  // Station says not raining but probability is missing — uncertain bucket.
  const d = pred({ station_raining_now: false });
  delete d.probability_pct;
  d.rain_category = "incert";
  assert.equal(verdictText(d, FRESH_NOW), "🌫️ Sense dades suficients");
});

test("missing probability_pct never produces 'undefined%' string", () => {
  const d = { timestamp: FRESH_TS };
  const out = verdictText(d, FRESH_NOW);
  assert.ok(!out.includes("undefined"), `Got: ${out}`);
  assert.ok(!out.includes("null"), `Got: ${out}`);
});
