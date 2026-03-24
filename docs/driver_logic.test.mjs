import test from "node:test";
import assert from "node:assert/strict";
import { explainGroup, selectDriverExplanations, getConceptTags } from "./driver_logic.js";

// ── Helpers ──────────────────────────────────────────────────────────

/** Default context: all null (unknown). Override specific fields as needed. */
function ctx(overrides = {}) {
  return {
    cloud: null, rh: null, pressChange: null, solar: null,
    radarKm: null, radarCov: null, rainAccum: null,
    cape: null, li: null, lightning: null, ensemble: {},
    ...overrides,
  };
}

/** Build a minimal prediction data object for selectDriverExplanations. */
function predData({ pct = 50, drivers = [], fv = {}, ensemble = {} } = {}) {
  return {
    probability_pct: pct,
    top_drivers: drivers,
    feature_vector: fv,
    ensemble,
  };
}

function driver(group, direction, contribution, icon = '🔹') {
  return { group, direction, contribution, icon };
}

// =====================================================================
// explainGroup — exhaustive per-group tests
// =====================================================================

test("Models globals — rain always returns the same string", () => {
  assert.equal(
    explainGroup("Models globals", "pluja", ctx()),
    "Les condicions meteorològiques afavoreixen pluja"
  );
});

test("Models globals — dry: sunny + dry air path", () => {
  assert.equal(
    explainGroup("Models globals", "sec", ctx({ solar: 300, rh: 40 })),
    "Fa sol i l'aire és sec"
  );
});

test("Models globals — dry: sunny but humid → skip to cloud check", () => {
  // solar>=200 but rh>=60, so first condition fails
  const result = explainGroup("Models globals", "sec", ctx({ solar: 300, rh: 65, cloud: 20 }));
  assert.equal(result, "Cel clar, sense senyals de pluja");
});

test("Models globals — dry: clear sky (low cloud)", () => {
  assert.equal(
    explainGroup("Models globals", "sec", ctx({ cloud: 10 })),
    "Cel clar, sense senyals de pluja"
  );
});

test("Models globals — dry: low humidity (no solar, cloud>=30)", () => {
  assert.equal(
    explainGroup("Models globals", "sec", ctx({ cloud: 50, rh: 40 })),
    "Aire sec, temps estable"
  );
});

test("Models globals — dry: fallback (all null)", () => {
  assert.equal(
    explainGroup("Models globals", "sec", ctx()),
    "Temps estable, sense senyals de pluja"
  );
});

test("Models globals — dry: fallback (high cloud, high rh, low solar)", () => {
  assert.equal(
    explainGroup("Models globals", "sec", ctx({ cloud: 80, rh: 70, solar: 50 })),
    "Temps estable, sense senyals de pluja"
  );
});

// ── Consistència NWP ──

test("Consistència NWP — rain", () => {
  assert.equal(explainGroup("Consistència NWP", "pluja", ctx()), "La situació de pluja és persistent");
});

test("Consistència NWP — dry → null (tautology)", () => {
  assert.equal(explainGroup("Consistència NWP", "sec", ctx()), null);
});

// ── Pluja confirmada ──

test("Pluja confirmada — rain with accumulation", () => {
  assert.equal(
    explainGroup("Pluja confirmada", "pluja", ctx({ rainAccum: 2.5 })),
    "Ja plou (2.5 mm en 3h)"
  );
});

test("Pluja confirmada — rain with zero accumulation → null", () => {
  assert.equal(explainGroup("Pluja confirmada", "pluja", ctx({ rainAccum: 0 })), null);
});

test("Pluja confirmada — rain with null accumulation → null", () => {
  // null > 0 is false in JS
  assert.equal(explainGroup("Pluja confirmada", "pluja", ctx({ rainAccum: null })), null);
});

test("Pluja confirmada — dry → null (always tautology)", () => {
  assert.equal(explainGroup("Pluja confirmada", "sec", ctx({ rainAccum: 5 })), null);
});

// ── Radar ──

test("Radar — rain with coverage > 0", () => {
  assert.equal(
    explainGroup("Radar", "pluja", ctx({ radarCov: 0.05 })),
    "Detectem pluja a prop"
  );
});

test("Radar — rain with no coverage but nearby echo", () => {
  assert.equal(
    explainGroup("Radar", "pluja", ctx({ radarCov: 0, radarKm: 15 })),
    "Detectem pluja a 15 km"
  );
});

test("Radar — rain with echo at exactly 25 km → null (not < 25)", () => {
  assert.equal(
    explainGroup("Radar", "pluja", ctx({ radarCov: 0, radarKm: 25 })),
    null
  );
});

test("Radar — rain with no data → null", () => {
  assert.equal(explainGroup("Radar", "pluja", ctx()), null);
});

test("Radar — dry → null (always skip)", () => {
  assert.equal(explainGroup("Radar", "sec", ctx({ radarCov: 0.1 })), null);
});

// ── Humitat ──

test("Humitat — rain with known rh", () => {
  assert.equal(explainGroup("Humitat", "pluja", ctx({ rh: 82.3 })), "L'aire és humit (82%)");
});

test("Humitat — rain with null rh", () => {
  assert.equal(explainGroup("Humitat", "pluja", ctx()), "L'aire és humit");
});

test("Humitat — dry with known rh", () => {
  assert.equal(explainGroup("Humitat", "sec", ctx({ rh: 35 })), "L'aire és sec (35%)");
});

test("Humitat — dry with null rh", () => {
  assert.equal(explainGroup("Humitat", "sec", ctx()), "L'aire és sec");
});

// ── Aigua precipitable ──

test("Aigua precipitable — rain", () => {
  assert.equal(explainGroup("Aigua precipitable", "pluja", ctx()), "Hi ha molta humitat a l'atmosfera");
});

test("Aigua precipitable — dry", () => {
  assert.equal(explainGroup("Aigua precipitable", "sec", ctx()), "Poca humitat a l'atmosfera");
});

// ── Inestabilitat ──

test("Inestabilitat — rain with high CAPE", () => {
  assert.equal(
    explainGroup("Inestabilitat", "pluja", ctx({ cape: 500 })),
    "L'atmosfera és inestable"
  );
});

test("Inestabilitat — rain with negative LI", () => {
  assert.equal(
    explainGroup("Inestabilitat", "pluja", ctx({ li: -3 })),
    "L'atmosfera és inestable"
  );
});

test("Inestabilitat — rain with both CAPE and LI meeting thresholds", () => {
  assert.equal(
    explainGroup("Inestabilitat", "pluja", ctx({ cape: 400, li: -1 })),
    "L'atmosfera és inestable"
  );
});

test("Inestabilitat — rain but CAPE too low and LI positive → null", () => {
  assert.equal(explainGroup("Inestabilitat", "pluja", ctx({ cape: 100, li: 2 })), null);
});

test("Inestabilitat — rain with CAPE exactly 300 (threshold boundary)", () => {
  assert.equal(
    explainGroup("Inestabilitat", "pluja", ctx({ cape: 300 })),
    "L'atmosfera és inestable"
  );
});

test("Inestabilitat — rain with CAPE=299 and LI=0 → null", () => {
  assert.equal(explainGroup("Inestabilitat", "pluja", ctx({ cape: 299, li: 0 })), null);
});

test("Inestabilitat — rain with null data → null", () => {
  assert.equal(explainGroup("Inestabilitat", "pluja", ctx()), null);
});

test("Inestabilitat — dry always stable", () => {
  assert.equal(explainGroup("Inestabilitat", "sec", ctx()), "L'atmosfera és estable");
});

// ── Pressió ──

test("Pressió — rain with dropping pressure", () => {
  assert.equal(explainGroup("Pressió", "pluja", ctx({ pressChange: -3 })), "La pressió baixa");
});

test("Pressió — rain with zero change (=0 is <=0)", () => {
  assert.equal(explainGroup("Pressió", "pluja", ctx({ pressChange: 0 })), "La pressió baixa");
});

test("Pressió — rain with rising pressure → null (contradiction)", () => {
  assert.equal(explainGroup("Pressió", "pluja", ctx({ pressChange: 2 })), null);
});

test("Pressió — rain with null pressure", () => {
  assert.equal(explainGroup("Pressió", "pluja", ctx()), "La pressió baixa");
});

test("Pressió — dry with stable/rising pressure", () => {
  assert.equal(explainGroup("Pressió", "sec", ctx({ pressChange: 1 })), "La pressió puja o és estable");
});

test("Pressió — dry with zero change", () => {
  assert.equal(explainGroup("Pressió", "sec", ctx({ pressChange: 0 })), "La pressió puja o és estable");
});

test("Pressió — dry with dropping pressure → null (contradiction)", () => {
  assert.equal(explainGroup("Pressió", "sec", ctx({ pressChange: -1 })), null);
});

test("Pressió — dry with null pressure", () => {
  assert.equal(explainGroup("Pressió", "sec", ctx()), "La pressió és estable");
});

// ── Règim de vent ──

test("Règim de vent — rain", () => {
  assert.equal(explainGroup("Règim de vent", "pluja", ctx()), "El vent porta humitat del mar");
});

test("Règim de vent — dry", () => {
  assert.equal(explainGroup("Règim de vent", "sec", ctx()), "El vent no porta humitat");
});

// ── Vent ──

test("Vent — rain", () => {
  assert.equal(explainGroup("Vent", "pluja", ctx()), "El vent afavoreix pluja");
});

test("Vent — dry", () => {
  assert.equal(explainGroup("Vent", "sec", ctx()), "El vent no afavoreix pluja");
});

// ── Núvols ──

test("Núvols — rain with high cloud", () => {
  assert.equal(explainGroup("Núvols", "pluja", ctx({ cloud: 85 })), "Cel ennuvolat (85%)");
});

test("Núvols — rain with exactly 50% cloud (threshold boundary)", () => {
  assert.equal(explainGroup("Núvols", "pluja", ctx({ cloud: 50 })), "Cel ennuvolat (50%)");
});

test("Núvols — rain with low cloud → null (contradiction)", () => {
  assert.equal(explainGroup("Núvols", "pluja", ctx({ cloud: 30 })), null);
});

test("Núvols — rain with null cloud", () => {
  assert.equal(explainGroup("Núvols", "pluja", ctx()), "Cel ennuvolat");
});

test("Núvols — dry with low cloud", () => {
  assert.equal(explainGroup("Núvols", "sec", ctx({ cloud: 20 })), "Cel obert (20% núvols)");
});

test("Núvols — dry with high cloud (still won't rain)", () => {
  assert.equal(
    explainGroup("Núvols", "sec", ctx({ cloud: 70 })),
    "Cel ennuvolat (70%), però no plourà"
  );
});

test("Núvols — dry with null cloud", () => {
  assert.equal(explainGroup("Núvols", "sec", ctx()), "Cel obert");
});

// ── Temperatura ──

test("Temperatura — rain", () => {
  assert.equal(explainGroup("Temperatura", "pluja", ctx()), "La temperatura afavoreix pluja");
});

test("Temperatura — dry → null (vacuous)", () => {
  assert.equal(explainGroup("Temperatura", "sec", ctx()), null);
});

// ── Hora del dia ──

test("Hora del dia — rain", () => {
  assert.equal(explainGroup("Hora del dia", "pluja", ctx()), "Hora propensa a pluja");
});

test("Hora del dia — dry → null (vacuous)", () => {
  assert.equal(explainGroup("Hora del dia", "sec", ctx()), null);
});

// ── Radiació solar ──

test("Radiació solar — rain with low solar", () => {
  assert.equal(
    explainGroup("Radiació solar", "pluja", ctx({ solar: 100 })),
    "Cel cobert, poca llum solar"
  );
});

test("Radiació solar — rain with high solar → null (contradiction)", () => {
  assert.equal(explainGroup("Radiació solar", "pluja", ctx({ solar: 500 })), null);
});

test("Radiació solar — rain with solar exactly 200 (threshold) → null", () => {
  assert.equal(explainGroup("Radiació solar", "pluja", ctx({ solar: 200 })), null);
});

test("Radiació solar — rain with null solar", () => {
  assert.equal(explainGroup("Radiació solar", "pluja", ctx()), "Poca llum solar");
});

test("Radiació solar — dry with high solar", () => {
  assert.equal(explainGroup("Radiació solar", "sec", ctx({ solar: 400 })), "Fa sol");
});

test("Radiació solar — dry with low solar → null", () => {
  assert.equal(explainGroup("Radiació solar", "sec", ctx({ solar: 50 })), null);
});

test("Radiació solar — dry with null solar", () => {
  assert.equal(explainGroup("Radiació solar", "sec", ctx()), "Fa sol");
});

// ── Sòl ──

test("Sòl — rain", () => {
  assert.equal(explainGroup("Sòl", "pluja", ctx()), "El terra està humit");
});

test("Sòl — dry → null", () => {
  assert.equal(explainGroup("Sòl", "sec", ctx()), null);
});

// ── Capa límit ──

test("Capa límit — rain", () => {
  assert.equal(explainGroup("Capa límit", "pluja", ctx()), "L'aire es barreja i pot generar xàfecs");
});

test("Capa límit — dry → null", () => {
  assert.equal(explainGroup("Capa límit", "sec", ctx()), null);
});

// ── Llamps ──

test("Llamps — rain with lightning detected", () => {
  assert.equal(explainGroup("Llamps", "pluja", ctx({ lightning: 7 })), "Detectem 7 llamps a prop");
});

test("Llamps — rain with zero lightning → null", () => {
  assert.equal(explainGroup("Llamps", "pluja", ctx({ lightning: 0 })), null);
});

test("Llamps — rain with null lightning → null", () => {
  assert.equal(explainGroup("Llamps", "pluja", ctx()), null);
});

test("Llamps — dry → null (always skip)", () => {
  assert.equal(explainGroup("Llamps", "sec", ctx({ lightning: 5 })), null);
});

// ── Sentinella ──

test("Sentinella — rain", () => {
  assert.equal(explainGroup("Sentinella", "pluja", ctx()), "Ja plou a localitats properes");
});

test("Sentinella — dry → null", () => {
  assert.equal(explainGroup("Sentinella", "sec", ctx()), null);
});

// ── Previsió oficial ──

test("Previsió oficial — rain → null (always skip)", () => {
  assert.equal(explainGroup("Previsió oficial", "pluja", ctx()), null);
});

test("Previsió oficial — dry → null (always skip)", () => {
  assert.equal(explainGroup("Previsió oficial", "sec", ctx()), null);
});

// ── Acord entre models ──

test("Acord entre models — rain with model count", () => {
  assert.equal(
    explainGroup("Acord entre models", "pluja", ctx({ ensemble: { models_rain: 3, total_models: 4 } })),
    "3 de 4 fonts independents coincideixen"
  );
});

test("Acord entre models — rain with model count, default total", () => {
  assert.equal(
    explainGroup("Acord entre models", "pluja", ctx({ ensemble: { models_rain: 2 } })),
    "2 de 4 fonts independents coincideixen"
  );
});

test("Acord entre models — rain without model count (fallback)", () => {
  assert.equal(
    explainGroup("Acord entre models", "pluja", ctx()),
    "Diverses fonts independents coincideixen"
  );
});

test("Acord entre models — dry → null (always skip)", () => {
  assert.equal(
    explainGroup("Acord entre models", "sec", ctx({ ensemble: { models_rain: 3 } })),
    null
  );
});

// ── Correcció local ──

test("Correcció local — rain", () => {
  assert.equal(
    explainGroup("Correcció local", "pluja", ctx()),
    "L'experiència local a Cardedeu ho confirma"
  );
});

test("Correcció local — dry", () => {
  assert.equal(
    explainGroup("Correcció local", "sec", ctx()),
    "L'experiència local a Cardedeu no hi dona suport"
  );
});

// ── Unknown group (default case) ──

test("Unknown group → null", () => {
  assert.equal(explainGroup("Grup Inventat", "pluja", ctx()), null);
  assert.equal(explainGroup("Grup Inventat", "sec", ctx()), null);
});

// =====================================================================
// selectDriverExplanations — selection logic tests
// =====================================================================

test("selectDriverExplanations — returns empty for no drivers", () => {
  const d = predData({ pct: 50, drivers: [] });
  assert.deepEqual(selectDriverExplanations(d), []);
});

test("selectDriverExplanations — returns empty for null drivers", () => {
  const d = predData({ pct: 50 });
  d.top_drivers = null;
  assert.deepEqual(selectDriverExplanations(d), []);
});

test("selectDriverExplanations — filters out Base (climatologia)", () => {
  const d = predData({
    pct: 50,
    drivers: [driver("Base (climatologia)", "sec", -2)],
  });
  assert.deepEqual(selectDriverExplanations(d), []);
});

test("selectDriverExplanations — extreme low (<10%) shows up to 3 dry only", () => {
  const d = predData({
    pct: 5,
    drivers: [
      driver("Models globals", "sec", -3, "🌤️"),
      driver("Humitat", "sec", -2, "💧"),
      driver("Pressió", "sec", -1, "📊"),
      driver("Vent", "sec", -0.5, "💨"),
      // Rain drivers should be ignored at extreme low
      driver("Radar", "pluja", 0.1, "📡"),
    ],
  });
  const results = selectDriverExplanations(d);
  assert.equal(results.length, 3);
  results.forEach(r => assert.equal(r.direction, "sec"));
});

test("selectDriverExplanations — extreme high (>90%) shows up to 3 rain only", () => {
  const d = predData({
    pct: 95,
    drivers: [
      driver("Radar", "pluja", 3, "📡"),
      driver("Humitat", "pluja", 2, "💧"),
      driver("Inestabilitat", "pluja", 1, "⚡"),
      driver("Vent", "pluja", 0.5, "💨"),
      // Dry drivers should be ignored at extreme high
      driver("Models globals", "sec", -1, "🌤️"),
    ],
    fv: { radarCov: 0.1, relative_humidity_2m: 85, cape: 500 },
  });
  const results = selectDriverExplanations(d);
  assert.equal(results.length, 3);
  results.forEach(r => assert.equal(r.direction, "pluja"));
});

test("selectDriverExplanations — middle zone shows 2 rain + 2 dry", () => {
  const d = predData({
    pct: 50,
    drivers: [
      driver("Humitat", "pluja", 2, "💧"),
      driver("Vent", "pluja", 1, "💨"),
      driver("Models globals", "sec", -2, "🌤️"),
      driver("Pressió", "sec", -1, "📊"),
    ],
  });
  const results = selectDriverExplanations(d);
  const rain = results.filter(r => r.direction === "pluja");
  const dry = results.filter(r => r.direction === "sec");
  assert.equal(rain.length, 2);
  assert.equal(dry.length, 2);
});

test("selectDriverExplanations — boundary: pct=10 is NOT extreme", () => {
  const d = predData({
    pct: 10,
    drivers: [
      driver("Humitat", "pluja", 2, "💧"),
      driver("Vent", "pluja", 1, "💨"),
      driver("Models globals", "sec", -2, "🌤️"),
      driver("Pressió", "sec", -1, "📊"),
    ],
  });
  const results = selectDriverExplanations(d);
  // Middle zone (10 is not < 10)
  const rain = results.filter(r => r.direction === "pluja");
  const dry = results.filter(r => r.direction === "sec");
  assert.ok(rain.length <= 2);
  assert.ok(dry.length <= 2);
});

test("selectDriverExplanations — boundary: pct=90 is NOT extreme", () => {
  const d = predData({
    pct: 90,
    drivers: [
      driver("Humitat", "pluja", 2, "💧"),
      driver("Vent", "pluja", 1, "💨"),
      driver("Models globals", "sec", -2, "🌤️"),
      driver("Pressió", "sec", -1, "📊"),
    ],
  });
  const results = selectDriverExplanations(d);
  // Middle zone (90 is not > 90)
  const rain = results.filter(r => r.direction === "pluja");
  const dry = results.filter(r => r.direction === "sec");
  assert.ok(rain.length <= 2);
  assert.ok(dry.length <= 2);
});

test("selectDriverExplanations — boundary: pct=9 IS extreme low", () => {
  const d = predData({
    pct: 9,
    drivers: [
      driver("Models globals", "sec", -3, "🌤️"),
      driver("Humitat", "sec", -2, "💧"),
      driver("Pressió", "sec", -1, "📊"),
    ],
  });
  const results = selectDriverExplanations(d);
  results.forEach(r => assert.equal(r.direction, "sec"));
});

test("selectDriverExplanations — boundary: pct=91 IS extreme high", () => {
  const d = predData({
    pct: 91,
    drivers: [
      driver("Humitat", "pluja", 3, "💧"),
      driver("Vent", "pluja", 2, "💨"),
      driver("Temperatura", "pluja", 1, "🌡️"),
    ],
  });
  const results = selectDriverExplanations(d);
  assert.equal(results.length, 3);
  results.forEach(r => assert.equal(r.direction, "pluja"));
});

test("selectDriverExplanations — null-returning groups are skipped, next one fills the slot", () => {
  // Inestabilitat with no CAPE/LI → null, so Vent should take the 2nd rain slot
  const d = predData({
    pct: 50,
    drivers: [
      driver("Humitat", "pluja", 3, "💧"),
      driver("Inestabilitat", "pluja", 2, "⚡"),
      driver("Vent", "pluja", 1, "💨"),
      driver("Models globals", "sec", -2, "🌤️"),
      driver("Pressió", "sec", -1, "📊"),
    ],
  });
  const results = selectDriverExplanations(d);
  const rain = results.filter(r => r.direction === "pluja");
  assert.equal(rain.length, 2);
  assert.ok(rain[0].text.includes("humit")); // Humitat
  assert.ok(rain[1].text.includes("vent")); // Vent (Inestabilitat was skipped)
});

test("selectDriverExplanations — all groups return null → empty result", () => {
  // Previsió oficial always returns null
  const d = predData({
    pct: 50,
    drivers: [
      driver("Previsió oficial", "pluja", 2, "📋"),
      driver("Previsió oficial", "sec", -1, "📋"),
    ],
  });
  assert.deepEqual(selectDriverExplanations(d), []);
});

test("selectDriverExplanations — extreme high skips null groups (Radar with no data)", () => {
  const d = predData({
    pct: 95,
    drivers: [
      driver("Radar", "pluja", 3, "📡"),         // null (no radar data)
      driver("Llamps", "pluja", 2, "⚡"),         // null (no lightning)
      driver("Humitat", "pluja", 1.5, "💧"),      // will produce text
      driver("Vent", "pluja", 1, "💨"),           // will produce text
      driver("Temperatura", "pluja", 0.5, "🌡️"), // will produce text
    ],
  });
  const results = selectDriverExplanations(d);
  assert.equal(results.length, 3);
  // First two should be skipped (null), so we get Humitat, Vent, Temperatura
  assert.ok(results[0].text.includes("humit"));
  assert.ok(results[1].text.includes("vent"));
  assert.ok(results[2].text.includes("temperatura"));
});

test("selectDriverExplanations — rain drivers sorted by contribution (highest first)", () => {
  const d = predData({
    pct: 95,
    drivers: [
      driver("Temperatura", "pluja", 1, "🌡️"),
      driver("Humitat", "pluja", 3, "💧"),
      driver("Vent", "pluja", 2, "💨"),
    ],
  });
  const results = selectDriverExplanations(d);
  // Humitat (3) > Vent (2) > Temperatura (1)
  assert.ok(results[0].text.includes("humit"));
  assert.ok(results[1].text.includes("vent"));
  assert.ok(results[2].text.includes("temperatura"));
});

test("selectDriverExplanations — dry drivers sorted by contribution (most negative first)", () => {
  const d = predData({
    pct: 5,
    drivers: [
      driver("Pressió", "sec", -1, "📊"),
      driver("Models globals", "sec", -3, "🌤️"),
      driver("Humitat", "sec", -2, "💧"),
    ],
  });
  const results = selectDriverExplanations(d);
  // Most negative first: Models globals (-3) > Humitat (-2) > Pressió (-1)
  assert.ok(results[0].text.includes("estable")); // Models globals
  assert.ok(results[1].text.includes("sec")); // Humitat
  assert.ok(results[2].text.includes("pressió")); // Pressió
});

test("selectDriverExplanations — feature_vector values passed correctly to explainGroup", () => {
  const d = predData({
    pct: 50,
    drivers: [
      driver("Pluja confirmada", "pluja", 3, "🌧️"),
      driver("Radar", "pluja", 2, "📡"),
      driver("Models globals", "sec", -2, "🌤️"),
      driver("Pressió", "sec", -1, "📊"),
    ],
    fv: {
      rain_accum_3h: 1.5,
      radar_coverage_20km: 0.08,
      shortwave_radiation: 350,
      relative_humidity_2m: 45,
      pressure_change_3h: 1.5,
    },
  });
  const results = selectDriverExplanations(d);
  const rain = results.filter(r => r.direction === "pluja");
  const dry = results.filter(r => r.direction === "sec");
  assert.ok(rain[0].text.includes("1.5 mm")); // Pluja confirmada uses rainAccum
  assert.ok(rain[1].text.includes("Detectem pluja a prop")); // Radar uses radarCov
  assert.ok(dry[0].text.includes("sol")); // Models globals: solar>=200, rh<60
  assert.ok(dry[1].text.includes("pressió")); // Pressió uses pressChange
});

test("selectDriverExplanations — ensemble data passed correctly", () => {
  const d = predData({
    pct: 95,
    drivers: [
      driver("Acord entre models", "pluja", 3, "🤝"),
      driver("Humitat", "pluja", 2, "💧"),
      driver("Vent", "pluja", 1, "💨"),
    ],
    ensemble: { models_rain: 3, total_models: 4 },
  });
  const results = selectDriverExplanations(d);
  assert.equal(results[0].text, "3 de 4 fonts independents coincideixen");
});

test("selectDriverExplanations — preserves icon from driver", () => {
  const d = predData({
    pct: 50,
    drivers: [
      driver("Humitat", "pluja", 2, "💧"),
      driver("Models globals", "sec", -2, "🌤️"),
    ],
  });
  const results = selectDriverExplanations(d);
  assert.equal(results[0].icon, "💧");
  assert.equal(results[1].icon, "🌤️");
});

test("selectDriverExplanations — middle zone with fewer than 2 rain drivers", () => {
  const d = predData({
    pct: 50,
    drivers: [
      driver("Vent", "pluja", 1, "💨"),
      driver("Models globals", "sec", -3, "🌤️"),
      driver("Humitat", "sec", -2, "💧"),
      driver("Pressió", "sec", -1, "📊"),
    ],
  });
  const results = selectDriverExplanations(d);
  const rain = results.filter(r => r.direction === "pluja");
  const dry = results.filter(r => r.direction === "sec");
  assert.equal(rain.length, 1); // Only 1 rain driver available
  assert.equal(dry.length, 2); // 2 dry as allowed
});

test("selectDriverExplanations — extreme low with fewer than 3 dry explanations available", () => {
  const d = predData({
    pct: 3,
    drivers: [
      driver("Models globals", "sec", -3, "🌤️"),
      // Only 1 explainable dry driver
    ],
  });
  const results = selectDriverExplanations(d);
  assert.equal(results.length, 1);
  assert.equal(results[0].direction, "sec");
});

// ── Data contradiction edge cases in selection context ──

test("selectDriverExplanations — pressure contradiction filtered during selection", () => {
  // Pressió pushing toward rain, but pressChange is positive → null, so it's skipped
  const d = predData({
    pct: 95,
    drivers: [
      driver("Pressió", "pluja", 3, "📊"),
      driver("Humitat", "pluja", 2, "💧"),
      driver("Vent", "pluja", 1, "💨"),
    ],
    fv: { pressure_change_3h: 2, relative_humidity_2m: 80 },
  });
  const results = selectDriverExplanations(d);
  // Pressió should be skipped (contradiction), Humitat and Vent fill in
  assert.equal(results.length, 2); // Only Humitat + Vent (no 3rd rain driver)
  assert.ok(results[0].text.includes("humit"));
  assert.ok(results[1].text.includes("vent"));
});

test("selectDriverExplanations — cloud contradiction at rain direction", () => {
  const d = predData({
    pct: 50,
    drivers: [
      driver("Núvols", "pluja", 2, "☁️"),
      driver("Humitat", "pluja", 1, "💧"),
      driver("Models globals", "sec", -2, "🌤️"),
      driver("Vent", "sec", -1, "💨"),
    ],
    fv: { cloud_cover: 20 }, // Low clouds → Núvols pluja returns null
  });
  const results = selectDriverExplanations(d);
  const rain = results.filter(r => r.direction === "pluja");
  // Núvols skipped, only Humitat available
  assert.equal(rain.length, 1);
  assert.ok(rain[0].text.includes("humit"));
});

// =====================================================================
// getConceptTags — concept tagging tests
// =====================================================================

test("getConceptTags — Models globals sec: 'Fa sol i l'aire és sec' → sunshine + dry_air", () => {
  const tags = getConceptTags("Models globals", "sec", "Fa sol i l'aire és sec");
  assert.ok(tags.includes("sunshine"));
  assert.ok(tags.includes("dry_air"));
  assert.equal(tags.length, 2);
});

test("getConceptTags — Models globals sec: 'Cel clar...' → sunshine", () => {
  const tags = getConceptTags("Models globals", "sec", "Cel clar, sense senyals de pluja");
  assert.deepEqual(tags, ["sunshine"]);
});

test("getConceptTags — Models globals sec: 'Aire sec, temps estable' → dry_air + stable", () => {
  const tags = getConceptTags("Models globals", "sec", "Aire sec, temps estable");
  assert.ok(tags.includes("dry_air"));
  assert.ok(tags.includes("stable"));
  assert.equal(tags.length, 2);
});

test("getConceptTags — Models globals sec: 'Temps estable...' → stable", () => {
  const tags = getConceptTags("Models globals", "sec", "Temps estable, sense senyals de pluja");
  assert.deepEqual(tags, ["stable"]);
});

test("getConceptTags — Models globals pluja → no concepts (unique)", () => {
  assert.deepEqual(getConceptTags("Models globals", "pluja", "Les condicions..."), []);
});

test("getConceptTags — Humitat sec → dry_air", () => {
  assert.deepEqual(getConceptTags("Humitat", "sec", "any"), ["dry_air"]);
});

test("getConceptTags — Humitat pluja → humidity", () => {
  assert.deepEqual(getConceptTags("Humitat", "pluja", "any"), ["humidity"]);
});

test("getConceptTags — Aigua precipitable sec → dry_air", () => {
  assert.deepEqual(getConceptTags("Aigua precipitable", "sec", "any"), ["dry_air"]);
});

test("getConceptTags — Aigua precipitable pluja → humidity", () => {
  assert.deepEqual(getConceptTags("Aigua precipitable", "pluja", "any"), ["humidity"]);
});

test("getConceptTags — Núvols sec → sunshine", () => {
  assert.deepEqual(getConceptTags("Núvols", "sec", "any"), ["sunshine"]);
});

test("getConceptTags — Núvols pluja → cloudy", () => {
  assert.deepEqual(getConceptTags("Núvols", "pluja", "any"), ["cloudy"]);
});

test("getConceptTags — Radiació solar sec → sunshine", () => {
  assert.deepEqual(getConceptTags("Radiació solar", "sec", "any"), ["sunshine"]);
});

test("getConceptTags — Radiació solar pluja → cloudy", () => {
  assert.deepEqual(getConceptTags("Radiació solar", "pluja", "any"), ["cloudy"]);
});

test("getConceptTags — Vent pluja → wind", () => {
  assert.deepEqual(getConceptTags("Vent", "pluja", "any"), ["wind"]);
});

test("getConceptTags — Règim de vent pluja → wind", () => {
  assert.deepEqual(getConceptTags("Règim de vent", "pluja", "any"), ["wind"]);
});

test("getConceptTags — Inestabilitat sec → stable", () => {
  assert.deepEqual(getConceptTags("Inestabilitat", "sec", "any"), ["stable"]);
});

test("getConceptTags — Pluja confirmada pluja → raining_nearby", () => {
  assert.deepEqual(getConceptTags("Pluja confirmada", "pluja", "any"), ["raining_nearby"]);
});

test("getConceptTags — Sentinella pluja → raining_nearby", () => {
  assert.deepEqual(getConceptTags("Sentinella", "pluja", "any"), ["raining_nearby"]);
});

test("getConceptTags — untagged groups return empty array", () => {
  assert.deepEqual(getConceptTags("Pressió", "pluja", "any"), []);
  assert.deepEqual(getConceptTags("Pressió", "sec", "any"), []);
  assert.deepEqual(getConceptTags("Temperatura", "pluja", "any"), []);
  assert.deepEqual(getConceptTags("Llamps", "pluja", "any"), []);
  assert.deepEqual(getConceptTags("Correcció local", "sec", "any"), []);
  assert.deepEqual(getConceptTags("Acord entre models", "pluja", "any"), []);
});

// =====================================================================
// Concept deduplication — integration tests
// =====================================================================

test("DEDUP: Models globals 'sec' + Humitat 'sec' → Humitat skipped (dry_air overlap)", () => {
  // The exact user scenario: both lines say "l'aire és sec"
  const d = predData({
    pct: 5,
    drivers: [
      driver("Models globals", "sec", -3, "🌐"),
      driver("Humitat", "sec", -2, "💧"),
      driver("Pressió", "sec", -1, "📊"),
    ],
    fv: { shortwave_radiation: 300, relative_humidity_2m: 42 },
  });
  const results = selectDriverExplanations(d);
  // Models globals: "Fa sol i l'aire és sec" → claims [sunshine, dry_air]
  // Humitat: "L'aire és sec (42%)" → wants [dry_air] → SKIPPED
  // Pressió: no concepts → allowed
  assert.equal(results.length, 2);
  assert.ok(results[0].text.includes("sol")); // Models globals
  assert.ok(results[1].text.includes("pressió")); // Pressió (Humitat was deduped)
});

test("DEDUP: Humitat first + Models globals second → Models globals shows but sunshine not deduped", () => {
  // Humitat appears first (higher contribution), claims dry_air
  // Models globals appears second with "Cel clar" → claims sunshine, no overlap
  const d = predData({
    pct: 5,
    drivers: [
      driver("Humitat", "sec", -3, "💧"),
      driver("Models globals", "sec", -2, "🌐"),
      driver("Pressió", "sec", -1, "📊"),
    ],
    fv: { cloud_cover: 20 }, // Models globals → "Cel clar, sense senyals de pluja" (sunshine, not dry_air)
  });
  const results = selectDriverExplanations(d);
  assert.equal(results.length, 3);
  assert.ok(results[0].text.includes("sec")); // Humitat → dry_air
  assert.ok(results[1].text.includes("Cel clar")); // Models globals → sunshine (no overlap)
  assert.ok(results[2].text.includes("pressió")); // Pressió
});

test("DEDUP: Vent + Règim de vent on same side → second skipped (wind overlap)", () => {
  const d = predData({
    pct: 95,
    drivers: [
      driver("Humitat", "pluja", 3, "💧"),
      driver("Vent", "pluja", 2, "💨"),
      driver("Règim de vent", "pluja", 1, "🌬️"),
      driver("Temperatura", "pluja", 0.5, "🌡️"),
    ],
  });
  const results = selectDriverExplanations(d);
  assert.equal(results.length, 3);
  assert.ok(results[0].text.includes("humit"));
  assert.ok(results[1].text.includes("vent")); // Vent (higher contrib)
  assert.ok(results[2].text.includes("temperatura")); // Règim de vent was skipped
});

test("DEDUP: Núvols sec + Radiació solar sec → second skipped (sunshine overlap)", () => {
  const d = predData({
    pct: 5,
    drivers: [
      driver("Núvols", "sec", -3, "☁️"),
      driver("Radiació solar", "sec", -2, "☀️"),
      driver("Pressió", "sec", -1, "📊"),
    ],
    fv: { cloud_cover: 15, shortwave_radiation: 400 },
  });
  const results = selectDriverExplanations(d);
  assert.equal(results.length, 2);
  assert.ok(results[0].text.includes("Cel obert")); // Núvols → sunshine
  // Radiació solar "Fa sol" → sunshine → SKIPPED
  assert.ok(results[1].text.includes("pressió")); // Pressió
});

test("DEDUP: Humitat pluja + Aigua precipitable pluja → second skipped (humidity overlap)", () => {
  const d = predData({
    pct: 95,
    drivers: [
      driver("Humitat", "pluja", 3, "💧"),
      driver("Aigua precipitable", "pluja", 2, "🌊"),
      driver("Vent", "pluja", 1, "💨"),
    ],
  });
  const results = selectDriverExplanations(d);
  assert.equal(results.length, 2);
  assert.ok(results[0].text.includes("humit")); // Humitat → humidity
  // Aigua precipitable → humidity → SKIPPED
  assert.ok(results[1].text.includes("vent")); // Vent
});

test("DEDUP: Núvols pluja + Radiació solar pluja → second skipped (cloudy overlap)", () => {
  const d = predData({
    pct: 50,
    drivers: [
      driver("Núvols", "pluja", 2, "☁️"),
      driver("Radiació solar", "pluja", 1, "☀️"),
      driver("Models globals", "sec", -2, "🌤️"),
      driver("Pressió", "sec", -1, "📊"),
    ],
    fv: { cloud_cover: 80, shortwave_radiation: 50 },
  });
  const results = selectDriverExplanations(d);
  const rain = results.filter(r => r.direction === "pluja");
  assert.equal(rain.length, 1); // Radiació solar skipped (cloudy overlap)
  assert.ok(rain[0].text.includes("ennuvolat")); // Núvols
});

test("DEDUP: Pluja confirmada + Sentinella → second skipped (raining_nearby overlap)", () => {
  const d = predData({
    pct: 95,
    drivers: [
      driver("Pluja confirmada", "pluja", 3, "🌧️"),
      driver("Sentinella", "pluja", 2, "🏔️"),
      driver("Humitat", "pluja", 1, "💧"),
    ],
    fv: { rain_accum_3h: 1.2 },
  });
  const results = selectDriverExplanations(d);
  assert.equal(results.length, 2);
  assert.ok(results[0].text.includes("Ja plou (1.2 mm")); // Pluja confirmada
  // Sentinella "Ja plou a localitats properes" → SKIPPED
  assert.ok(results[1].text.includes("humit")); // Humitat
});

test("DEDUP: Models globals 'estable' + Inestabilitat sec → Inestabilitat skipped (stable overlap)", () => {
  const d = predData({
    pct: 5,
    drivers: [
      driver("Models globals", "sec", -3, "🌐"),
      driver("Inestabilitat", "sec", -2, "⚡"),
      driver("Pressió", "sec", -1, "📊"),
    ],
    // All null → Models globals returns "Temps estable, sense senyals de pluja" → ['stable']
  });
  const results = selectDriverExplanations(d);
  assert.equal(results.length, 2);
  assert.ok(results[0].text.includes("estable")); // Models globals
  // Inestabilitat "L'atmosfera és estable" → ['stable'] → SKIPPED
  assert.ok(results[1].text.includes("pressió")); // Pressió
});

test("DEDUP: cross-direction concepts are shared (rain humidity blocks dry dry_air)", () => {
  // In middle zone, rain is processed first. If Humitat pluja claims 'humidity',
  // Aigua precipitable sec claims 'dry_air' — these are DIFFERENT concepts → no block
  const d = predData({
    pct: 50,
    drivers: [
      driver("Humitat", "pluja", 2, "💧"),
      driver("Vent", "pluja", 1, "💨"),
      driver("Aigua precipitable", "sec", -2, "🌊"),
      driver("Pressió", "sec", -1, "📊"),
    ],
  });
  const results = selectDriverExplanations(d);
  // humidity ≠ dry_air → no cross-direction dedup
  assert.equal(results.length, 4);
});

test("DEDUP: groups with no concepts never get deduped", () => {
  const d = predData({
    pct: 5,
    drivers: [
      driver("Pressió", "sec", -3, "📊"),
      driver("Correcció local", "sec", -2, "📍"),
      driver("Models globals", "sec", -1, "🌐"),
    ],
  });
  const results = selectDriverExplanations(d);
  // All have unique/empty concepts → all shown
  assert.equal(results.length, 3);
});
