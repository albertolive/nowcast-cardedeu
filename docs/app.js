const REPO = 'albertolive/nowcast-cardedeu';
const BRANCH = 'main';
const RAW_BASE = `https://raw.githubusercontent.com/${REPO}/${BRANCH}/data`;
const REFRESH_INTERVAL_MS = 5 * 60 * 1000; // 5 minutes

async function fetchJSON(filename) {
  // Try local (GitHub Pages docs/) first, fall back to raw.githubusercontent.com
  for (const base of ['.', RAW_BASE]) {
    try {
      const r = await fetch(`${base}/${filename}`, { cache: 'no-cache' });
      if (r.ok) return r.json();
    } catch {}
  }
  throw new Error(`No s'ha pogut carregar ${filename}`);
}

async function fetchJSONL(filename) {
  for (const base of ['.', RAW_BASE]) {
    try {
      const r = await fetch(`${base}/${filename}`, { cache: 'no-cache' });
      if (r.ok) {
        const text = await r.text();
        return text.trim().split('\n').filter(Boolean).map(line => JSON.parse(line));
      }
    } catch {}
  }
  throw new Error(`No s'ha pogut carregar ${filename}`);
}

function fmtTime(iso) {
  const d = new Date(iso);
  return d.toLocaleString('ca-ES', {
    day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit'
  });
}

function fmtTimeShort(iso) {
  const d = new Date(iso);
  return d.toLocaleString('ca-ES', { hour: '2-digit', minute: '2-digit' });
}

function getProbColor(pct) {
  if (pct >= 60) return 'var(--accent-blue)';
  if (pct >= 35) return 'var(--accent-yellow)';
  return 'var(--accent-green)';
}

function relativeTime(isoStr) {
  const diff = Date.now() - new Date(isoStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'ara mateix';
  if (mins === 1) return 'fa 1 minut';
  if (mins < 60) return `fa ${mins} minuts`;
  const hours = Math.floor(mins / 60);
  if (hours === 1) return 'fa 1 hora';
  if (hours < 24) return `fa ${hours} hores`;
  return `fa ${Math.floor(hours / 24)} dies`;
}

function compassLabel(deg) {
  if (deg == null) return '—';
  const dirs = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSO','SO','OSO','O','ONO','NO','NNO'];
  return dirs[Math.round(deg / 22.5) % 16];
}

function renderPrediction(latest, history) {
  const pct = latest.probability_pct;
  const noPct = (100 - pct).toFixed(1);
  const color = getProbColor(pct);
  const circumference = 2 * Math.PI * 88;
  const offset = circumference * (1 - pct / 100);
  const confColor = pct >= 60 ? 'conf-high' : pct >= 35 ? 'conf-medium' : 'conf-low';
  const gateOpen = latest.rain_gate_open;

  // Verified history stats (prediction-level)
  const verified = history.filter(h => h.verified);
  const correct = verified.filter(h => h.correct).length;
  const accuracy = verified.length > 0 ? ((correct / verified.length) * 100).toFixed(0) : '—';

  // Day-level accuracy: a day is "correct" if majority of verified predictions were right
  const dayBuckets = {};
  for (const h of history) {
    if (!h.verified) continue;
    const day = h.timestamp.slice(0, 10);
    if (!dayBuckets[day]) dayBuckets[day] = { correct: 0, total: 0 };
    dayBuckets[day].total++;
    if (h.correct) dayBuckets[day].correct++;
  }
  const dayKeys = Object.keys(dayBuckets);
  const daysCorrect = dayKeys.filter(d => dayBuckets[d].correct / dayBuckets[d].total >= 0.5).length;
  const daysTotal = dayKeys.length;
  const dayAccuracy = daysTotal > 0 ? ((daysCorrect / daysTotal) * 100).toFixed(0) : '—';

  const app = document.getElementById('app');
  app.innerHTML = `
    <!-- Main prediction -->
    <div class="prediction-card">
      <div class="prediction-question">Plourà a Cardedeu en la propera hora?</div>
      <div class="prediction-subtext">${fmtTime(latest.timestamp)} · <span id="freshness">${relativeTime(latest.timestamp)}</span></div>

      <div class="probability-ring">
        <svg viewBox="0 0 200 200" width="200" height="200">
          <circle class="ring-bg" cx="100" cy="100" r="88"/>
          <circle class="ring-fill" cx="100" cy="100" r="88"
            stroke="${color}"
            stroke-dasharray="${circumference}"
            stroke-dashoffset="${offset}"/>
        </svg>
        <div class="probability-value">
          <div class="pct" style="color:${color}">${pct}%</div>
          <div class="label">probabilitat</div>
          <div class="conf ${confColor}">${latest.confidence}</div>
        </div>
      </div>

      <div class="outcome-buttons">
        <div class="outcome-btn yes ${latest.will_rain ? 'active' : ''}">
          <span class="btn-label">🌧️ Sí, plourà</span>
          <span class="btn-pct">${pct}%</span>
        </div>
        <div class="outcome-btn no ${!latest.will_rain ? 'active' : ''}">
          <span class="btn-label">☀️ No plourà</span>
          <span class="btn-pct">${noPct}%</span>
        </div>
      </div>

      <div class="gate-indicator ${gateOpen ? 'open' : 'closed'}">
        <span class="gate-dot"></span>
        ${gateOpen ? '🌧️ Senyals de pluja detectats — radar, llamps o previsions oficials actius' : '☀️ Cap senyal de pluja — radar, llamps i previsions oficials en calma'}
      </div>

      <div class="prediction-meta">
        Encerts per dia: <strong style="color:var(--accent-green)">${dayAccuracy}%</strong> (${daysCorrect}/${daysTotal} dies)
        · Per predicció: <strong>${accuracy}%</strong> (${correct}/${verified.length})
      </div>
    </div>

    <!-- Probability chart -->
    <div class="chart-card">
      <h2>📈 Com ha canviat la probabilitat (últimes 24h)</h2>
      <p class="chart-hint">Toca o passa el ratolí per veure els valors</p>
      <div class="chart-container">
        <canvas id="probChart"></canvas>
      </div>
    </div>

    <div class="grid">
      <!-- Conditions -->
      <div class="info-card">
        <h3>🌡️ Condicions actuals</h3>
        ${renderConditions(latest)}
      </div>

      <!-- Radar -->
      <div class="info-card">
        <h3>📡 Radar</h3>
        ${renderRadar(latest)}
      </div>

      <!-- Atmospheric -->
      <div class="info-card">
        <h3>🌀 Atmosfera</h3>
        ${renderAtmosphere(latest)}
      </div>

      <!-- Why this prediction -->
      <div class="info-card">
        <h3>🧠 Per què aquesta predicció?</h3>
        ${renderWhyPrediction(latest)}
      </div>
    </div>

    <!-- Sources bar -->
    <div class="sources-bar">
      <span style="font-weight:600">Fonts:</span>
      ${renderSources(latest)}
    </div>

    <!-- Resolution History -->
    <div class="history-card">
      <h2>📋 Encerts i errors</h2>
      <p class="history-subtitle">
        Comprovem cada predicció amb la pluja que realment va caure. Clica un dia per veure el detall.
      </p>
      <div id="calendar-root"></div>
    </div>
  `;

  initCalendar(history);
  drawChart(history, latest);
}

function renderConditions(d) {
  const c = d.conditions || {};
  const fv = d.feature_vector || {};
  const pressChange = d.pressure_change_3h;
  const pressArrow = pressChange != null ? (pressChange > 0.5 ? '↑' : pressChange < -0.5 ? '↓' : '→') : '';
  const pressColor = pressChange != null && pressChange < -1.5 ? 'color:var(--accent-red)' : pressChange != null && pressChange > 1.5 ? 'color:var(--accent-blue)' : '';
  const pressTrend = pressChange != null ? ` ${pressArrow}(${pressChange > 0 ? '+' : ''}${pressChange.toFixed(1)}/3h)` : '';
  const dewPoint = fv.dew_point != null ? fv.dew_point.toFixed(1) + '°C' : '—';
  const solar = c.solar_radiation;
  const solarDesc = solar != null ? (solar > 600 ? '☀️ Intens' : solar > 300 ? '🌤️ Moderat' : solar > 50 ? '🌥️ Baix' : '🌑 Nul') : '';
  return `
    <div class="stat-row"><span class="stat-label">Temperatura</span><span class="stat-value">${c.temperature || '—'}°C</span></div>
    <div class="stat-row"><span class="stat-label">Humitat</span><span class="stat-value">${c.humidity || '—'}%</span></div>
    <div class="stat-row"><span class="stat-label">Punt de rosada</span><span class="stat-value">${dewPoint}</span></div>
    <div class="stat-row"><span class="stat-label">Pressió</span><span class="stat-value" style="${pressColor}">${c.pressure || '—'} hPa${pressTrend}</span></div>
    <div class="stat-row"><span class="stat-label">Vent</span><span class="stat-value">${c.wind_speed || '—'} km/h ${c.wind_dir || ''}</span></div>
    <div class="stat-row"><span class="stat-label">Radiació solar</span><span class="stat-value">${solar != null ? Math.round(solar) + ' W/m² ' + solarDesc : '—'}</span></div>
    <div class="stat-row"><span class="stat-label">Pluja avui</span><span class="stat-value">${c.rain_today || '0.0'} mm</span></div>
  `;
}

function renderRadar(d) {
  const r = d.radar || {};
  const q = r.quadrants || {};
  // Build mini compass showing which quadrants have echoes
  const quadLabels = ['N','E','S','W'];
  const compassParts = quadLabels.map(dir => {
    const dbz = q[`max_dbz_${dir}`] || 0;
    const cov = q[`coverage_${dir}`] || 0;
    if (dbz > 5) return `<span style="color:var(--accent-blue);font-weight:700">${dir}</span>`;
    return `<span style="color:var(--text-muted);opacity:0.3">${dir}</span>`;
  }).join(' · ');
  const hasQuadrants = quadLabels.some(dir => (q[`max_dbz_${dir}`] || 0) > 5);
  return `
    <div class="stat-row"><span class="stat-label">Pluja més propera</span><span class="stat-value">${r.nearest_echo_km != null && r.nearest_echo_km < 30 ? r.nearest_echo_km + ' km ' + (r.nearest_echo_compass || '') : 'No detectada'}</span></div>
    <div class="stat-row"><span class="stat-label">Zona amb pluja</span><span class="stat-value">${r.coverage_20km != null ? r.coverage_20km + '% (radi 20 km)' : '—'}</span></div>
    <div class="stat-row"><span class="stat-label">S'acosta?</span><span class="stat-value">${r.approaching ? '⚠️ Sí' : 'No'}${r.storm_eta_min ? ' (~' + r.storm_eta_min + ' min)' : ''}</span></div>
    <div class="stat-row"><span class="stat-label">Direcció</span><span class="stat-value">${hasQuadrants ? compassParts : '<span style="color:var(--text-muted)">Sense pluja al radar</span>'}</span></div>
    <div class="stat-row"><span class="stat-label">Intensitat</span><span class="stat-value">${r.dbz != null && r.dbz > 0 ? (r.dbz >= 40 ? '🟥 Forta' : r.dbz >= 25 ? '🟨 Moderada' : '🟩 Feble') : 'Res detectat'}</span></div>
    <p class="card-hint">Escaneig cada 10 min en un radi de 30 km al voltant de Cardedeu</p>
  `;
}

function renderSources(d) {
  const fv = d.feature_vector || {};
  const sources = [
    { name: 'Estació local', active: d.conditions?.temperature != null },
    { name: 'Radar RainViewer', active: d.radar?.dbz != null },
    { name: 'Llamps XDDE', active: d.rain_gate_open },
    { name: 'Meteocat XEMA', active: d.sentinel?.temp != null || d.sentinel?.humidity != null },
    { name: 'AEMET', active: d.aemet?.prob_precip != null },
    { name: 'Ensemble NWP', active: (d.ensemble?.total_models || 0) > 0 },
  ];
  return sources.map(s => `
    <span class="source-item">
      <span class="dot ${s.active ? 'active' : 'inactive'}"></span>
      ${s.name}
    </span>
  `).join('');
}

function renderWhyPrediction(d) {
  const r = d.radar || {};
  const e = d.ensemble || {};
  const a = d.aemet || {};
  const p = d.pressure_levels || {};

  // Radar
  let radarText, radarColor;
  if (r.approaching) {
    radarText = `⚠️ Pluja a ${r.nearest_echo_km || '?'} km, acostant-se`;
    radarColor = 'var(--accent-red)';
  } else if (r.has_echo || r.dbz > 5) {
    radarText = `Ecos a ${r.nearest_echo_km || '?'} km, estàtica`;
    radarColor = 'var(--accent-yellow)';
  } else {
    radarText = 'Sense pluja';
    radarColor = 'var(--accent-green)';
  }

  // NWP consensus
  const modelsRain = e.models_rain || 0;
  const totalModels = e.total_models || 4;
  let modelsText, modelsColor;
  if (modelsRain === 0) {
    modelsText = `0/${totalModels} preveuen pluja`;
    modelsColor = 'var(--accent-green)';
  } else if (modelsRain <= totalModels / 2) {
    modelsText = `${modelsRain}/${totalModels} preveuen pluja`;
    modelsColor = 'var(--accent-yellow)';
  } else {
    modelsText = `${modelsRain}/${totalModels} preveuen pluja`;
    modelsColor = 'var(--accent-red)';
  }

  // AEMET
  const aemetProb = a.prob_precip ?? null;
  let aemetText = '—', aemetColor = 'var(--text-muted)';
  if (aemetProb != null) {
    aemetText = aemetProb + '%';
    aemetColor = aemetProb >= 50 ? 'var(--accent-red)' : aemetProb >= 20 ? 'var(--accent-yellow)' : 'var(--accent-green)';
  }

  // AEMET storm
  const stormProb = a.prob_storm ?? null;
  let stormText = '—', stormColor = 'var(--text-muted)';
  if (stormProb != null) {
    stormText = stormProb + '%';
    stormColor = stormProb >= 40 ? 'var(--accent-red)' : stormProb >= 15 ? 'var(--accent-yellow)' : 'var(--accent-green)';
  }

  const detailId = 'why-detail-' + Date.now();
  return `
    <div class="stat-row"><span class="stat-label">Radar</span><span class="stat-value" style="color:${radarColor}">${radarText}</span></div>
    <div class="stat-row"><span class="stat-label">Models globals</span><span class="stat-value" style="color:${modelsColor}">${modelsText}</span></div>
    <div class="stat-row"><span class="stat-label">Previsió AEMET</span><span class="stat-value" style="color:${aemetColor}">${aemetText}</span></div>
    <div class="stat-row"><span class="stat-label">Prob. tempesta</span><span class="stat-value" style="color:${stormColor}">${stormText}</span></div>

    <button class="expand-toggle" onclick="this.classList.toggle('open');document.getElementById('${detailId}').classList.toggle('open')">
      <span class="chevron">▶</span> Com s'ha calculat
    </button>
    <div id="${detailId}" class="expand-content">
      <p class="tech-explainer">
        El sistema combina ${d.features_used || '210'} variables meteorològiques — estació local, radar, llamps, 4 models globals i 12 anys d'històric de Cardedeu — per corregir els errors dels models globals al nostre microclima.
      </p>
      <p class="tech-explainer" style="margin-top:8px">
        ${d.raw_probability != null && d.threshold ? `El model genera un ${(d.raw_probability * 100).toFixed(1)}% inicial, que es calibra a <strong>${d.probability_pct}%</strong> (probabilitat real basada en l'històric). Com que ${d.probability_pct}% ${d.probability_pct >= d.threshold * 100 ? '≥' : '<'} ${(d.threshold * 100).toFixed(0)}% (llindar), el veredicte és: <strong>${d.will_rain ? '🌧️ plourà' : '☀️ no plourà'}</strong>.` : `La probabilitat calibrada és <strong>${d.probability_pct}%</strong>.`}
        El model es re-entrena cada dia amb les prediccions verificades.
      </p>
    </div>
  `;
}

function renderAtmosphere(d) {
  const p = d.pressure_levels || {};
  const w = d.wind_regime || {};
  const regimes = {
    llevantada: { icon: '🌊', name: 'Llevantada', desc: 'Humitat del mar contra les muntanyes — pluja #1 a Cardedeu (15% de probabilitat)' },
    tramuntana: { icon: '❄️', name: 'Tramuntana', desc: 'Vent fred del nord — supressor de pluja (5%)' },
    migjorn: { icon: '🌡️', name: 'Migjorn', desc: 'Aire càlid africà — segon en pluja (15%)' },
    garbi: { icon: '🌀', name: 'Garbí', desc: 'Aire inestable del sud-oest — tempestes (11%)' },
    ponent: { icon: '🏔️', name: 'Ponent', desc: 'Aire sec continental — supressor de pluja (6%)' },
  };
  const activeRegime = w.is_llevantada ? regimes.llevantada :
                       w.is_tramuntana ? regimes.tramuntana :
                       w.is_migjorn ? regimes.migjorn :
                       w.is_garbi ? regimes.garbi :
                       w.is_ponent ? regimes.ponent : null;
  const regime = activeRegime ? `${activeRegime.icon} ${activeRegime.name}` : '🧭 Neutre';
  const regimeDesc = activeRegime ? activeRegime.desc : 'Sense règim dominant';

  // Human-readable stability assessment from indices
  let stabilityText = '—';
  let stabilityColor = 'var(--text-muted)';
  if (p.li_index != null) {
    if (p.li_index < -6) { stabilityText = '⛈️ Molt inestable'; stabilityColor = 'var(--accent-red)'; }
    else if (p.li_index < -3) { stabilityText = '🌩️ Inestable'; stabilityColor = 'var(--accent-yellow)'; }
    else if (p.li_index < 0) { stabilityText = '⚠️ Lleugerament inestable'; stabilityColor = 'var(--accent-yellow)'; }
    else if (p.li_index < 3) { stabilityText = '🌤️ Estable'; stabilityColor = 'var(--accent-green)'; }
    else { stabilityText = '☀️ Molt estable'; stabilityColor = 'var(--accent-green)'; }
  } else if (p.tt_index != null) {
    if (p.tt_index > 55) { stabilityText = '⛈️ Molt inestable'; stabilityColor = 'var(--accent-red)'; }
    else if (p.tt_index > 50) { stabilityText = '🌩️ Inestable'; stabilityColor = 'var(--accent-yellow)'; }
    else if (p.tt_index > 44) { stabilityText = '⚠️ Lleugerament inestable'; stabilityColor = 'var(--accent-yellow)'; }
    else { stabilityText = '☀️ Estable'; stabilityColor = 'var(--accent-green)'; }
  }

  // Human-readable humidity at altitude
  let humidityText = '—';
  if (p.rh_700 != null) {
    if (p.rh_700 > 80) humidityText = '💧 Molt humit';
    else if (p.rh_700 > 60) humidityText = '🌥️ Humit';
    else if (p.rh_700 > 40) humidityText = '⛅ Moderat';
    else humidityText = '☀️ Sec';
  }

  // Jet stream interpretation
  let jetText = '—';
  if (p.wind_300_speed_kmh != null) {
    if (p.wind_300_speed_kmh > 120) jetText = '💨 Molt fort (' + Math.round(p.wind_300_speed_kmh) + ' km/h)';
    else if (p.wind_300_speed_kmh > 80) jetText = '🌬️ Fort (' + Math.round(p.wind_300_speed_kmh) + ' km/h)';
    else if (p.wind_300_speed_kmh > 40) jetText = 'Moderat (' + Math.round(p.wind_300_speed_kmh) + ' km/h)';
    else jetText = 'Feble (' + Math.round(p.wind_300_speed_kmh) + ' km/h)';
  }

  // 850hPa wind details
  const wind850Dir = p.wind_850_dir ?? (p.wind_850_speed_kmh != null ? w.wind_dir : null);
  const wind850Speed = p.wind_850_speed_kmh;
  const wind850Text = wind850Speed != null ? `${compassLabel(wind850Dir)} · ${Math.round(wind850Speed)} km/h` : '—';

  // SST
  const sst = d.sst?.sst_med;
  let sstText = '—';
  if (sst != null) {
    sstText = sst.toFixed(1) + '°C';
    if (sst >= 25) sstText += ' 🔥 Molt càlid';
    else if (sst >= 20) sstText += ' 🌊 Càlid';
    else if (sst >= 15) sstText += ' 🌡️ Temperat';
    else sstText += ' ❄️ Fred';
  }

  const detailId = 'atmo-detail-' + Date.now();
  return `
    <div class="stat-row"><span class="stat-label">Tipus de vent</span><span class="stat-value">${regime}</span></div>
    <div class="regime-desc">${regimeDesc}</div>
    <div class="stat-row"><span class="stat-label">Vent sinòptic (850hPa)</span><span class="stat-value">${wind850Text}</span></div>
    <div class="stat-row"><span class="stat-label">Risc de tempesta</span><span class="stat-value" style="color:${stabilityColor}">${stabilityText}</span></div>
    <div class="stat-row"><span class="stat-label">Humitat en altura</span><span class="stat-value">${humidityText}</span></div>
    <div class="stat-row"><span class="stat-label">Vent a gran altitud</span><span class="stat-value">${jetText}</span></div>
    <div class="stat-row"><span class="stat-label">Mar Mediterrani</span><span class="stat-value">${sstText}</span></div>

    <button class="expand-toggle" onclick="this.classList.toggle('open');document.getElementById('${detailId}').classList.toggle('open')">
      <span class="chevron">▶</span> Detalls per capes de l'atmosfera
    </button>
    <div id="${detailId}" class="expand-content">
      <div class="stat-row"><span class="stat-label">925 hPa · ~750m</span><span class="stat-value">${p.temp_925 ?? '—'}°C · ${p.rh_925 ?? '—'}%</span></div>
      <div class="stat-row"><span class="stat-label">850 hPa · ~1.500m</span><span class="stat-value">${p.temp_850 ?? '—'}°C · ${p.rh_850 ?? '—'}%</span></div>
      <div class="stat-row"><span class="stat-label">700 hPa · ~3.000m</span><span class="stat-value">${p.temp_700 ?? '—'}°C · ${p.rh_700 ?? '—'}%</span></div>
      <div class="stat-row"><span class="stat-label">500 hPa · ~5.500m</span><span class="stat-value">${p.temp_500 ?? '—'}°C</span></div>
      <div class="stat-row"><span class="stat-label">300 hPa · ~9.000m</span><span class="stat-value">vent ${p.wind_300_speed_kmh != null ? Math.round(p.wind_300_speed_kmh) + ' km/h' : '—'}</span></div>
      <div class="stat-row"><span class="stat-label">Índex VT / TT</span><span class="stat-value">${p.vt_index != null ? p.vt_index.toFixed(1) : '—'} / ${p.tt_index != null ? p.tt_index.toFixed(1) : '—'}</span></div>
      <div class="stat-row"><span class="stat-label">Lifted Index</span><span class="stat-value">${p.li_index != null ? p.li_index.toFixed(1) : '—'}</span></div>
    </div>
  `;
}

/* ---- Calendar-based Resolution History ---- */
function initCalendar(history) {
  const root = document.getElementById('calendar-root');
  if (!root) return;

  // Group predictions by day
  const dayMap = {};
  for (const h of history) {
    const day = h.timestamp.slice(0, 10);
    if (!dayMap[day]) dayMap[day] = [];
    dayMap[day].push(h);
  }

  // Find date range
  const allDays = Object.keys(dayMap).sort();
  const today = new Date();
  let currentYear = today.getFullYear();
  let currentMonth = today.getMonth();
  let selectedDay = null;

  const WEEKDAYS = ['dl', 'dt', 'dc', 'dj', 'dv', 'ds', 'dg'];

  function getDaySummary(dayKey) {
    const preds = dayMap[dayKey];
    if (!preds) return null;
    const verified = preds.filter(p => p.verified);
    const correct = verified.filter(p => p.correct).length;
    const anyRain = preds.some(p => p.actual_rain === true);
    const allPending = verified.length === 0;
    const hasPending = preds.some(p => !p.verified);
    const rainMm = preds.reduce((max, p) => Math.max(max, p.actual_rain_mm || 0), 0);
    const acc = verified.length > 0 ? (correct / verified.length) : null;
    const todayStr = new Date().toISOString().slice(0, 10);
    const isOngoing = dayKey >= todayStr;
    return { preds, verified, correct, anyRain, allPending, hasPending, isOngoing, rainMm, acc, count: preds.length };
  }

  function render() {
    const year = currentYear;
    const month = currentMonth;
    const firstDay = new Date(year, month, 1);
    const lastDay = new Date(year, month + 1, 0);
    const daysInMonth = lastDay.getDate();
    // Monday=0 start
    let startWeekday = (firstDay.getDay() + 6) % 7;

    const monthLabel = firstDay.toLocaleDateString('ca-ES', { month: 'long', year: 'numeric' });

    // Month stats
    let mVerified = 0, mCorrect = 0, mPreds = 0, mRainDays = 0, mDaysCorrect = 0, mDaysWithData = 0;
    for (let d = 1; d <= daysInMonth; d++) {
      const key = `${year}-${String(month + 1).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
      const s = getDaySummary(key);
      if (s) {
        mPreds += s.count;
        mVerified += s.verified.length;
        mCorrect += s.correct;
        if (s.anyRain) mRainDays++;
        if (s.verified.length > 0) {
          mDaysWithData++;
          if (s.acc >= 0.5) mDaysCorrect++;
        }
      }
    }
    const mAcc = mVerified > 0 ? ((mCorrect / mVerified) * 100).toFixed(0) : '—';
    const mDayAcc = mDaysWithData > 0 ? ((mDaysCorrect / mDaysWithData) * 100).toFixed(0) : '—';

    // Can navigate?
    const earliest = allDays.length > 0 ? allDays[0] : `${year}-${String(month + 1).padStart(2, '0')}-01`;
    const earliestDate = new Date(earliest + 'T12:00:00');
    const canPrev = year > earliestDate.getFullYear() || (year === earliestDate.getFullYear() && month > earliestDate.getMonth());
    const canNext = year < today.getFullYear() || (year === today.getFullYear() && month < today.getMonth());

    // Build calendar cells
    let cells = '';
    // Weekday headers
    for (const wd of WEEKDAYS) {
      cells += `<div class="cal-weekday">${wd}</div>`;
    }
    // Empty cells before first day
    for (let i = 0; i < startWeekday; i++) {
      cells += `<div class="cal-day empty"></div>`;
    }
    // Day cells
    const todayKey = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`;
    for (let d = 1; d <= daysInMonth; d++) {
      const key = `${year}-${String(month + 1).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
      const s = getDaySummary(key);
      const isToday = key === todayKey;
      const isSelected = key === selectedDay;

      let cls = 'cal-day';
      let icon = '';
      if (!s) {
        cls += ' no-data';
      } else {
        cls += ' has-data';
        if (s.allPending) {
          cls += ' pending-day';
          icon = '⏳';
        } else if (s.isOngoing) {
          cls += ' pending-day';
          icon = s.anyRain ? '🌧️' : '📊';
        } else if (s.anyRain) {
          cls += ' rain-day';
          icon = '🌧️';
        } else if (s.acc >= 0.9) {
          cls += ' perfect';
          icon = '✓';
        } else if (s.acc >= 0.6) {
          cls += ' good';
          icon = '~';
        } else {
          cls += ' bad';
          icon = '✗';
        }
      }
      if (isToday) cls += ' today';
      if (isSelected) cls += ' selected';

      cells += `<div class="${cls}" data-day="${key}">
        <span class="cal-day-num">${d}</span>
        ${icon ? `<span class="cal-day-icon">${icon}</span>` : ''}
      </div>`;
    }

    let html = `
      <div class="cal-nav">
        <button class="cal-nav-btn" id="cal-prev" ${canPrev ? '' : 'disabled'}>← Anterior</button>
        <span class="cal-month-label">${monthLabel.charAt(0).toUpperCase() + monthLabel.slice(1)}</span>
        <button class="cal-nav-btn" id="cal-next" ${canNext ? '' : 'disabled'}>Següent →</button>
      </div>

      <div class="cal-month-stats">
        <span class="cal-stat">Dies encertats: <strong>${mDayAcc}%</strong> (${mDaysCorrect}/${mDaysWithData})</span>
        <span class="cal-stat">Prediccions: <strong>${mAcc}%</strong> (${mCorrect}/${mVerified})</span>
        <span class="cal-stat">Dies de pluja: <strong>${mRainDays}</strong></span>
      </div>

      <div class="cal-grid">${cells}</div>

      <div class="cal-legend">
        <span class="cal-legend-item"><span class="cal-legend-dot" style="background:rgba(63,185,80,0.25)"></span> Tot correcte</span>
        <span class="cal-legend-item"><span class="cal-legend-dot" style="background:rgba(88,166,255,0.2)"></span> Dia de pluja</span>
        <span class="cal-legend-item"><span class="cal-legend-dot" style="background:rgba(210,153,34,0.2)"></span> Pendent / En curs</span>
        <span class="cal-legend-item"><span class="cal-legend-dot" style="background:rgba(248,81,73,0.2)"></span> Errors</span>
        <span class="cal-legend-item"><span class="cal-legend-dot" style="background:var(--surface2)"></span> Sense dades</span>
      </div>

      <div id="day-detail-panel"></div>
    `;

    root.innerHTML = html;

    // If a day is selected, render its detail
    if (selectedDay && dayMap[selectedDay]) {
      renderDayDetail(selectedDay);
    }

    // Event listeners
    document.getElementById('cal-prev')?.addEventListener('click', () => {
      currentMonth--;
      if (currentMonth < 0) { currentMonth = 11; currentYear--; }
      render();
    });
    document.getElementById('cal-next')?.addEventListener('click', () => {
      currentMonth++;
      if (currentMonth > 11) { currentMonth = 0; currentYear++; }
      render();
    });
    root.querySelectorAll('.cal-day.has-data').forEach(el => {
      el.addEventListener('click', () => {
        const day = el.dataset.day;
        selectedDay = selectedDay === day ? null : day;
        render();
      });
    });
  }

  function renderDayDetail(dayKey) {
    const panel = document.getElementById('day-detail-panel');
    if (!panel) return;
    const s = getDaySummary(dayKey);
    if (!s) { panel.innerHTML = ''; return; }

    const d = new Date(dayKey + 'T12:00:00');
    const dayLabel = d.toLocaleDateString('ca-ES', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' });

    let resolutionCls, resolutionText;
    if (s.allPending) {
      resolutionCls = 'pending';
      resolutionText = '⏳ Pendent';
    } else if (s.isOngoing) {
      resolutionCls = 'in-progress';
      resolutionText = s.anyRain
        ? `🌧️ En curs — Ha plogut (${s.rainMm.toFixed(1)} mm)`
        : '📊 En curs — Encara no ha plogut';
    } else if (s.anyRain) {
      resolutionCls = 'rain';
      resolutionText = `🌧️ Va ploure (${s.rainMm.toFixed(1)} mm)`;
    } else {
      resolutionCls = 'no-rain';
      resolutionText = '☀️ No va ploure';
    }

    let accCls = '', accText = '';
    if (s.verified.length > 0) {
      const pct = ((s.correct / s.verified.length) * 100).toFixed(0);
      accCls = pct >= 90 ? 'perfect' : pct >= 60 ? 'good' : 'bad';
      accText = `${s.correct}/${s.verified.length} encerts`;
    }

    const predRows = s.preds.map(p => {
      const t = new Date(p.timestamp);
      const time = t.toLocaleString('ca-ES', { hour: '2-digit', minute: '2-digit' });
      const pct = p.probability_pct;
      const color = getProbColor(pct);
      const said = p.will_rain ? '🌧️ Plourà' : '☀️ No plourà';
      let resultText, resultCls;
      if (!p.verified) {
        resultText = '⏳ Pendent';
        resultCls = 'pending';
      } else if (p.correct) {
        resultText = '✅ Encert';
        resultCls = 'correct';
      } else {
        resultText = '❌ Error';
        resultCls = 'wrong';
      }
      const rainInfo = p.actual_rain_mm != null ? `${p.actual_rain_mm.toFixed(1)} mm` : '—';
      return `
        <div class="pred-row">
          <span class="pred-time">${time}</span>
          <div class="pred-prob-bar"><div class="pred-prob-fill" style="width:${pct}%;background:${color}"></div></div>
          <span class="pred-pct" style="color:${color}">${pct}%</span>
          <span class="pred-said">${said}</span>
          <span class="pred-result ${resultCls}">${resultText}</span>
          <span class="pred-rain-mm">${rainInfo}</span>
        </div>`;
    }).join('');

    const tableHeader = `
      <div class="pred-header">
        <span class="pred-time">Hora</span>
        <div class="pred-prob-bar"></div>
        <span class="pred-pct">Prob.</span>
        <span class="pred-said">Vam dir</span>
        <span class="pred-result">Resultat</span>
        <span class="pred-rain-mm">Pluja real</span>
      </div>`;

    panel.innerHTML = `
      <div class="day-detail">
        <div class="day-detail-header">
          <div class="day-detail-left">
            <span class="day-detail-date">${dayLabel.charAt(0).toUpperCase() + dayLabel.slice(1)}</span>
            <span class="day-resolution ${resolutionCls}">${resolutionText}</span>
          </div>
          <div style="display:flex;align-items:center;gap:10px">
            <span class="day-detail-accuracy ${accCls}">${accText}</span>
            <button class="day-detail-close" id="close-detail" title="Tancar">✕</button>
          </div>
        </div>
        ${tableHeader}
        ${predRows}
      </div>`;

    document.getElementById('close-detail')?.addEventListener('click', () => {
      selectedDay = null;
      render();
    });
  }

  render();
}

/* ---- Lightweight Chart (pure Canvas, no dependencies) ---- */
function drawChart(history, latest) {
  const canvas = document.getElementById('probChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;

  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width;
  const H = rect.height;

  // Data
  const now = Date.now();
  const cutoff = now - 24 * 60 * 60 * 1000;
  const points = history
    .filter(h => new Date(h.timestamp).getTime() > cutoff)
    .map(h => ({ t: new Date(h.timestamp).getTime(), p: h.probability_pct }));

  if (points.length < 2) {
    ctx.fillStyle = '#8b949e';
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Acumulant dades...', W / 2, H / 2);
    return;
  }

  const pad = { top: 20, right: 16, bottom: 32, left: 42 };
  const cW = W - pad.left - pad.right;
  const cH = H - pad.top - pad.bottom;

  const tMin = points[0].t;
  const tMax = points[points.length - 1].t;
  const tRange = tMax - tMin || 1;

  const xScale = t => pad.left + ((t - tMin) / tRange) * cW;
  const yScale = p => pad.top + cH - (p / 100) * cH;

  function paintChart(hoverIdx) {
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    // Grid lines
    ctx.strokeStyle = 'rgba(48,54,61,0.5)';
    ctx.lineWidth = 1;
    for (let pct of [0, 25, 50, 75, 100]) {
      const yy = yScale(pct);
      ctx.beginPath();
      ctx.moveTo(pad.left, yy);
      ctx.lineTo(W - pad.right, yy);
      ctx.stroke();

      ctx.fillStyle = '#8b949e';
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(pct + '%', pad.left - 6, yy + 4);
    }

    // Threshold line
    const threshY = yScale(latest.threshold ? latest.threshold * 100 : 40);
    ctx.strokeStyle = 'rgba(248,81,73,0.3)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(pad.left, threshY);
    ctx.lineTo(W - pad.right, threshY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Area fill
    const gradient = ctx.createLinearGradient(0, pad.top, 0, pad.top + cH);
    gradient.addColorStop(0, 'rgba(88,166,255,0.25)');
    gradient.addColorStop(1, 'rgba(88,166,255,0.02)');
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.moveTo(xScale(points[0].t), yScale(0));
    for (const pt of points) ctx.lineTo(xScale(pt.t), yScale(pt.p));
    ctx.lineTo(xScale(points[points.length - 1].t), yScale(0));
    ctx.closePath();
    ctx.fill();

    // Line
    ctx.strokeStyle = '#58a6ff';
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    for (let i = 0; i < points.length; i++) {
      const px = xScale(points[i].t);
      const py = yScale(points[i].p);
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Latest dot
    const last = points[points.length - 1];
    ctx.beginPath();
    ctx.arc(xScale(last.t), yScale(last.p), 5, 0, Math.PI * 2);
    ctx.fillStyle = '#58a6ff';
    ctx.fill();
    ctx.strokeStyle = '#0d1117';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Time labels
    ctx.fillStyle = '#8b949e';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    const maxLabels = Math.min(6, Math.floor(cW / 70));
    const minLabelGap = 60;
    let lastLabelX = -Infinity;
    for (let i = 0; i <= maxLabels; i++) {
      const t = tMin + (tRange * i) / maxLabels;
      const px = xScale(t);
      if (px - lastLabelX < minLabelGap) continue;
      const d = new Date(t);
      const h = String(d.getHours()).padStart(2, '0');
      const m = String(d.getMinutes()).padStart(2, '0');
      ctx.fillText(`${h}:${m}`, px, H - 8);
      lastLabelX = px;
    }

    // Threshold label
    ctx.fillStyle = 'rgba(248,81,73,0.5)';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('llindar', pad.left + 4, threshY - 4);

    // Hover crosshair + tooltip
    if (hoverIdx != null && hoverIdx >= 0 && hoverIdx < points.length) {
      const pt = points[hoverIdx];
      const hx = xScale(pt.t);
      const hy = yScale(pt.p);

      // Vertical line
      ctx.strokeStyle = 'rgba(230,237,243,0.3)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(hx, pad.top);
      ctx.lineTo(hx, pad.top + cH);
      ctx.stroke();
      ctx.setLineDash([]);

      // Horizontal line
      ctx.strokeStyle = 'rgba(230,237,243,0.2)';
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(pad.left, hy);
      ctx.lineTo(W - pad.right, hy);
      ctx.stroke();
      ctx.setLineDash([]);

      // Highlighted dot
      ctx.beginPath();
      ctx.arc(hx, hy, 6, 0, Math.PI * 2);
      ctx.fillStyle = '#58a6ff';
      ctx.fill();
      ctx.strokeStyle = '#e6edf3';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Tooltip
      const d = new Date(pt.t);
      const timeStr = d.toLocaleString('ca-ES', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
      const probStr = pt.p.toFixed(1) + '%';
      const tooltipText = `${timeStr}  ·  ${probStr}`;
      ctx.font = '600 12px -apple-system, BlinkMacSystemFont, sans-serif';
      const tw = ctx.measureText(tooltipText).width;
      const tPad = 8;
      const tH = 28;
      let tx = hx - (tw + tPad * 2) / 2;
      // Keep tooltip within bounds
      if (tx < pad.left) tx = pad.left;
      if (tx + tw + tPad * 2 > W - pad.right) tx = W - pad.right - tw - tPad * 2;
      let ty = hy - tH - 12;
      if (ty < 4) ty = hy + 12;

      // Tooltip background
      ctx.fillStyle = 'rgba(22,27,34,0.95)';
      ctx.beginPath();
      const cr = 6;
      ctx.moveTo(tx + cr, ty);
      ctx.lineTo(tx + tw + tPad * 2 - cr, ty);
      ctx.quadraticCurveTo(tx + tw + tPad * 2, ty, tx + tw + tPad * 2, ty + cr);
      ctx.lineTo(tx + tw + tPad * 2, ty + tH - cr);
      ctx.quadraticCurveTo(tx + tw + tPad * 2, ty + tH, tx + tw + tPad * 2 - cr, ty + tH);
      ctx.lineTo(tx + cr, ty + tH);
      ctx.quadraticCurveTo(tx, ty + tH, tx, ty + tH - cr);
      ctx.lineTo(tx, ty + cr);
      ctx.quadraticCurveTo(tx, ty, tx + cr, ty);
      ctx.closePath();
      ctx.fill();
      ctx.strokeStyle = 'rgba(48,54,61,0.8)';
      ctx.lineWidth = 1;
      ctx.stroke();

      // Tooltip text
      ctx.fillStyle = '#e6edf3';
      ctx.font = '600 12px -apple-system, BlinkMacSystemFont, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(tooltipText, tx + tPad, ty + tH / 2 + 4);
    }

    ctx.restore();
  }

  // Initial paint
  paintChart(null);

  // Find nearest point to a canvas-relative x coordinate
  function findNearest(canvasX) {
    let best = 0;
    let bestDist = Infinity;
    for (let i = 0; i < points.length; i++) {
      const dist = Math.abs(xScale(points[i].t) - canvasX);
      if (dist < bestDist) { bestDist = dist; best = i; }
    }
    return bestDist < 40 ? best : null;
  }

  function getCanvasX(e) {
    const r = canvas.getBoundingClientRect();
    if (e.touches && e.touches.length > 0) return e.touches[0].clientX - r.left;
    return e.clientX - r.left;
  }

  // Mouse events
  canvas.addEventListener('mousemove', e => {
    paintChart(findNearest(getCanvasX(e)));
    canvas.style.cursor = 'crosshair';
  });
  canvas.addEventListener('mouseleave', () => {
    paintChart(null);
    canvas.style.cursor = '';
  });

  // Touch events for mobile
  canvas.addEventListener('touchstart', e => {
    const idx = findNearest(getCanvasX(e));
    if (idx != null) {
      e.preventDefault();
      paintChart(idx);
    }
  }, { passive: false });
  canvas.addEventListener('touchmove', e => {
    const idx = findNearest(getCanvasX(e));
    if (idx != null) {
      e.preventDefault();
      paintChart(idx);
    }
  }, { passive: false });
  canvas.addEventListener('touchend', () => paintChart(null));
}

/* ---- Init ---- */
let _latestTimestamp = null;

async function loadAndRender() {
  const [latest, history] = await Promise.all([
    fetchJSON('latest_prediction.json'),
    fetchJSONL('predictions_log.jsonl')
  ]);
  const isUpdate = _latestTimestamp && _latestTimestamp !== latest.timestamp;
  _latestTimestamp = latest.timestamp;
  renderPrediction(latest, history);
  if (isUpdate) {
    const card = document.querySelector('.prediction-card');
    if (card) { card.classList.add('flash'); setTimeout(() => card.classList.remove('flash'), 1500); }
  }
  return { latest, history };
}

async function init() {
  try {
    const { latest, history } = await loadAndRender();

    // Resize chart on window resize
    let resizeTimer;
    window.addEventListener('resize', () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => drawChart(history, latest), 200);
    });

    // Auto-refresh every 5 minutes
    setInterval(async () => {
      try {
        const data = await loadAndRender();
        // Update resize handler
        window.removeEventListener('resize', null);
        window.addEventListener('resize', () => {
          clearTimeout(resizeTimer);
          resizeTimer = setTimeout(() => drawChart(data.history, data.latest), 200);
        });
      } catch (e) {
        console.warn('Auto-refresh failed:', e);
      }
    }, REFRESH_INTERVAL_MS);

    // Update "fa X minuts" every 30 seconds
    setInterval(() => {
      const el = document.getElementById('freshness');
      if (el && _latestTimestamp) el.textContent = relativeTime(_latestTimestamp);
    }, 30000);
  } catch (err) {
    document.getElementById('app').innerHTML = `
      <div class="prediction-card" style="color:var(--accent-red)">
        <p>Error carregant dades: ${err.message}</p>
        <p style="color:var(--text-muted);font-size:13px;margin-top:8px">
          Comprova que el repositori és públic o actualitza la URL.
        </p>
      </div>
    `;
  }
}

init();
