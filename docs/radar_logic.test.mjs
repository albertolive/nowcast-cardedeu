import test from "node:test";
import assert from "node:assert/strict";

import { deriveRadarViewModel } from "./radar_logic.js";

test("no echoes anywhere => no detection, 0% coverage, no intensity", () => {
  const model = deriveRadarViewModel({
    radar: {
      has_echo: false,
      nearest_echo_km: 30,
      max_dbz_20km: 0,
      coverage_20km: 0,
      quadrants: {},
    },
    feature_vector: {
      aemet_radar_has_echo: 0,
      aemet_radar_echoes_found: 0,
      aemet_radar_nearest_echo_km: 30,
      aemet_radar_max_dbz_20km: 0,
      aemet_radar_coverage_20km: 0,
    },
  });

  assert.equal(model.bestHasEcho, false);
  assert.equal(model.nearestText, "No detectada");
  assert.equal(model.coverageText, "0% (radi 20 km)");
  assert.equal(model.intensityText, "Res detectat");
});

test("contradiction guard: no detected echo but residual aemet metrics => still no detection", () => {
  const model = deriveRadarViewModel({
    radar: {
      has_echo: false,
      nearest_echo_km: 30,
      max_dbz_20km: 0,
      coverage_20km: 0,
      quadrants: {},
    },
    feature_vector: {
      aemet_radar_has_echo: 0,
      aemet_radar_echoes_found: 0,
      aemet_radar_nearest_echo_km: 6,
      aemet_radar_max_dbz_20km: 40,
      aemet_radar_coverage_20km: 0.0396,
    },
  });

  assert.equal(model.bestHasEcho, false);
  assert.equal(model.nearestText, "No detectada");
  assert.equal(model.coverageText, "0% (radi 20 km)");
  assert.equal(model.intensityText, "Res detectat");
});

test("aemet nearby echo area (not over Cardedeu) => detected and coherent", () => {
  const model = deriveRadarViewModel({
    radar: {
      has_echo: false,
      nearest_echo_km: 30,
      max_dbz_20km: 0,
      coverage_20km: 0,
      quadrants: {},
    },
    feature_vector: {
      aemet_radar_has_echo: 0,
      aemet_radar_echoes_found: 1,
      aemet_radar_nearest_echo_km: 6,
      aemet_radar_max_dbz_20km: 40,
      aemet_radar_coverage_20km: 0.0396,
    },
  });

  assert.equal(model.bestHasEcho, true);
  assert.equal(model.nearestText, "6 km");
  assert.equal(model.coverageText, "4% (radi 20 km)");
  assert.equal(model.intensityText, "Forta (40 dBZ)");
});

test("rainviewer echo => uses rainviewer metrics", () => {
  const model = deriveRadarViewModel({
    radar: {
      has_echo: false,
      nearest_echo_km: 8.5,
      nearest_echo_compass: "S",
      max_dbz_20km: 24,
      coverage_20km: 0.12,
      quadrants: { max_dbz_S: 24 },
    },
    feature_vector: {
      aemet_radar_has_echo: 0,
      aemet_radar_echoes_found: 0,
      aemet_radar_nearest_echo_km: 30,
      aemet_radar_max_dbz_20km: 0,
      aemet_radar_coverage_20km: 0,
    },
  });

  assert.equal(model.bestHasEcho, true);
  assert.equal(model.nearestText, "8.5 km S");
  assert.equal(model.coverageText, "12% (radi 20 km)");
  assert.equal(model.intensityText, "Feble (24 dBZ)");
});

// ─── Cross-field coherence tests ────────────────────────────────

test("COHERENCE: no echo => radarVoteRain false, gateSignalText null, radarVoteDetail says 'sense pluja'", () => {
  const m = deriveRadarViewModel({
    radar: { has_echo: false, nearest_echo_km: 30, max_dbz_20km: 0, coverage_20km: 0, quadrants: {} },
    feature_vector: { aemet_radar_has_echo: 0, aemet_radar_echoes_found: 0,
      aemet_radar_nearest_echo_km: 30, aemet_radar_max_dbz_20km: 0, aemet_radar_coverage_20km: 0 },
  });

  // All three rendering paths must agree: no rain detected
  assert.equal(m.bestHasEcho, false, "radar card: no echo");
  assert.equal(m.radarVoteRain, false, "why-prediction: no rain vote");
  assert.equal(m.gateSignalText, null, "gate signals: no radar chip");
  assert.match(m.radarVoteDetail, /[Ss]ense pluja/, "vote detail says no rain");
  assert.equal(m.nearestText, "No detectada", "nearest text consistent");
});

test("COHERENCE: residual AEMET metrics but no echo flags => all paths agree no rain", () => {
  const m = deriveRadarViewModel({
    radar: { has_echo: false, nearest_echo_km: 30, max_dbz_20km: 0, coverage_20km: 0, quadrants: {} },
    feature_vector: { aemet_radar_has_echo: 0, aemet_radar_echoes_found: 0,
      aemet_radar_nearest_echo_km: 6, aemet_radar_max_dbz_20km: 40, aemet_radar_coverage_20km: 0.04 },
  });

  // This is the exact bug scenario — UI must NOT show contradictions
  assert.equal(m.bestHasEcho, false);
  assert.equal(m.radarVoteRain, false);
  assert.equal(m.gateSignalText, null);
  assert.equal(m.bestCov, 0, "coverage gated to 0");
  assert.equal(m.bestDbz, 0, "dbz gated to 0");
});

test("COHERENCE: echo detected => all paths agree rain present", () => {
  const m = deriveRadarViewModel({
    radar: { has_echo: false, nearest_echo_km: 12, nearest_echo_compass: "NE",
      max_dbz_20km: 30, coverage_20km: 0.08, quadrants: {} },
    feature_vector: { aemet_radar_has_echo: 0, aemet_radar_echoes_found: 0,
      aemet_radar_nearest_echo_km: 30, aemet_radar_max_dbz_20km: 0, aemet_radar_coverage_20km: 0 },
  });

  assert.equal(m.bestHasEcho, true, "radar card: echo detected");
  assert.equal(m.radarVoteRain, true, "why-prediction: rain vote");
  assert.ok(m.gateSignalText, "gate signals: radar chip present");
  assert.match(m.gateSignalText, /12 km/, "gate shows distance");
  assert.notEqual(m.nearestText, "No detectada");
  assert.ok(m.bestDbz > 0, "dbz populated");
  assert.ok(m.bestCov > 0, "coverage populated");
});

test("COHERENCE: approaching storm => all paths reflect approach", () => {
  const m = deriveRadarViewModel({
    radar: { has_echo: true, approaching: true, nearest_echo_km: 5,
      nearest_echo_compass: "W", max_dbz_20km: 35, coverage_20km: 0.15, quadrants: {} },
    feature_vector: { aemet_radar_has_echo: 0, aemet_radar_echoes_found: 0,
      aemet_radar_nearest_echo_km: 30, aemet_radar_max_dbz_20km: 0, aemet_radar_coverage_20km: 0 },
  });

  assert.equal(m.approaching, true);
  assert.equal(m.radarVoteRain, true);
  assert.match(m.radarVoteDetail, /acostant-se/, "vote detail mentions approaching");
  assert.ok(m.gateSignalText, "gate chip present");
});

test("COHERENCE: both radars detect => vote mentions 2 radars", () => {
  const m = deriveRadarViewModel({
    radar: { has_echo: true, nearest_echo_km: 10, max_dbz_20km: 28, coverage_20km: 0.1, quadrants: {} },
    feature_vector: { aemet_radar_has_echo: 1, aemet_radar_echoes_found: 1,
      aemet_radar_nearest_echo_km: 8, aemet_radar_max_dbz_20km: 35, aemet_radar_coverage_20km: 0.05 },
  });

  assert.equal(m.rvHasEcho, true);
  assert.equal(m.aemetHasEcho, true);
  assert.equal(m.radarVoteRain, true);
  assert.match(m.radarVoteDetail, /2 radars/, "mentions both radars");
});

test("COHERENCE: only AEMET area echo => vote mentions AEMET", () => {
  const m = deriveRadarViewModel({
    radar: { has_echo: false, nearest_echo_km: 30, max_dbz_20km: 0, coverage_20km: 0, quadrants: {} },
    feature_vector: { aemet_radar_has_echo: 0, aemet_radar_echoes_found: 1,
      aemet_radar_nearest_echo_km: 15, aemet_radar_max_dbz_20km: 25, aemet_radar_coverage_20km: 0.02 },
  });

  assert.equal(m.aemetHasEcho, true);
  assert.equal(m.rvHasEcho, false);
  assert.equal(m.radarVoteRain, true);
  assert.match(m.radarVoteDetail, /AEMET/, "mentions AEMET source");
  assert.match(m.radarVoteDetail, /moderada/i, "correct intensity label for 25 dBZ");
});

test("COHERENCE: low dBZ below threshold => not detected", () => {
  const m = deriveRadarViewModel({
    radar: { has_echo: false, nearest_echo_km: 15, max_dbz_20km: 5, coverage_20km: 0.01, quadrants: {} },
    feature_vector: { aemet_radar_has_echo: 0, aemet_radar_echoes_found: 0,
      aemet_radar_nearest_echo_km: 30, aemet_radar_max_dbz_20km: 0, aemet_radar_coverage_20km: 0 },
  });

  // dBZ=5 below RADAR_MIN_DBZ=10 — should NOT count as echo
  assert.equal(m.rvHasEcho, false, "below min dBZ threshold");
  assert.equal(m.bestHasEcho, false);
  assert.equal(m.radarVoteRain, false);
  assert.equal(m.gateSignalText, null);
});
