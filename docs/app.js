const REPO = 'albertolive/nowcast-cardedeu';
const BRANCH = 'main';
const RAW_BASE = `https://raw.githubusercontent.com/${REPO}/${BRANCH}/data`;
const REFRESH_INTERVAL_MS = 5 * 60 * 1000; // 5 minutes

import { deriveRadarViewModel } from './radar_logic.js';
import { selectDriverExplanations, GROUP_TOOLTIP } from './driver_logic.js';

function getDataBases() {
  // Local docs/ first (fresh slim JSONL pushed every 10 min by the predict job),
  // then raw.githubusercontent as fallback (full JSONL).
  return ['.', RAW_BASE];
}

const DATA_BASES = getDataBases();

async function fetchJSON(filename) {
  for (const base of DATA_BASES) {
    try {
      const ctrl = new AbortController();
      const tid = setTimeout(() => ctrl.abort(), 8000);
      const r = await fetch(`${base}/${filename}`, { cache: 'no-cache', signal: ctrl.signal });
      clearTimeout(tid);
      if (r.ok) return r.json();
    } catch {}
  }
  throw new Error(`No s'ha pogut carregar ${filename}`);
}

async function fetchJSONL(filename) {
  for (const base of DATA_BASES) {
    try {
      const ctrl = new AbortController();
      const tid = setTimeout(() => ctrl.abort(), 8000);
      const r = await fetch(`${base}/${filename}`, { cache: 'no-cache', signal: ctrl.signal });
      clearTimeout(tid);
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

function _gateSignals(d) {
  const signals = [];
  const e = d.ensemble || {};
  const a = d.aemet || {};
  const fv = d.feature_vector || {};
  const radarView = deriveRadarViewModel(d);
  if (radarView.gateSignalText) {
    signals.push(radarView.gateSignalText);
  }
  if ((fv.lightning_count_30km || 0) > 0) signals.push('⚡ Llamps');
  const agreement = e.rain_agreement ?? (e.models_rain != null && e.total_models ? e.models_rain / e.total_models : 0);
  if (agreement >= 0.2) signals.push('🌐 ' + (e.models_rain || 0) + '/' + (e.total_models || 4) + ' models');
  if ((a.prob_storm || 0) >= 10) signals.push('⛈️ Tronada ' + a.prob_storm + '%');
  if ((fv.cape || 0) >= 800) signals.push('🔥 CAPE alt');
  return signals;
}

function _probTrend(history) {
  const now = Date.now();
  const recent = history
    .filter(h => now - new Date(h.timestamp).getTime() < 60 * 60 * 1000)
    .map(h => h.probability_pct);
  if (recent.length < 3) return { arrow: '', label: '', cls: '' };
  const first = recent.slice(0, Math.ceil(recent.length / 2));
  const last = recent.slice(Math.floor(recent.length / 2));
  const avg = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
  const diff = avg(last) - avg(first);
  if (diff > 5) return { arrow: '↑', label: 'pujant', cls: 'trend-up' };
  if (diff < -5) return { arrow: '↓', label: 'baixant', cls: 'trend-down' };
  return { arrow: '→', label: 'estable', cls: 'trend-stable' };
}

/** Honest verdict text based on rain_category or probability thresholds */
function _verdictText(d) {
  const cat = d.rain_category;
  const pct = d.probability_pct;
  if (cat === 'probable' || pct >= 65) return '🌧️ Pluja probable';
  if (cat === 'sec' || pct < 30) return '☀️ No plourà';
  // Uncertain zone: show the probability honestly
  return `🌤️ ${pct}% probabilitat de pluja`;
}

/** Display label for history rows: sec/incert/probable */
function _predictionLabel(d) {
  const cat = d.rain_category;
  const pct = d.probability_pct;
  if (cat === 'probable' || pct >= 65) return '🌧️ Pluja probable';
  if (cat === 'sec' || pct < 30) return '☀️ No plourà';
  return '🌤️ Incert';
}

/** Fair verification: uncertain zone scored softly with 50% lean boundary */
function _verificationResult(d) {
  if (!d.verified) return { text: '⏳ Pendent', cls: 'pending' };
  const cat = d.rain_category;
  const pct = d.probability_pct;
  const isUncertain = cat === 'incert' || (pct >= 30 && pct < 65 && cat == null);
  if (isUncertain) {
    // 50% = natural "which side did you lean?" boundary
    const leanedRain = pct >= 50;
    const wasRight = leanedRain === Boolean(d.actual_rain);
    return wasRight
      ? { text: '🔸 Encert', cls: 'uncertain' }
      : { text: '🔸 Error', cls: 'uncertain' };
  }
  // Retrocompat: if correct is null (old data verified before rain_category), recompute
  const correct = d.correct != null ? d.correct : (pct < 30 ? !d.actual_rain : d.actual_rain);
  if (correct) return { text: '✅ Encert', cls: 'correct' };
  return { text: '❌ Error', cls: 'wrong' };
}

function renderPrediction(latest, history) {
  const pct = latest.probability_pct;
  const noPct = (100 - pct).toFixed(1);
  const color = getProbColor(pct);
  const circumference = 2 * Math.PI * 88;
  const offset = circumference * (1 - pct / 100);
  const confColor = pct >= 60 ? 'conf-high' : pct >= 35 ? 'conf-medium' : 'conf-low';
  const gateOpen = latest.rain_gate_open;
  const signals = gateOpen ? _gateSignals(latest) : [];
  const trend = _probTrend(history);

  // Verified history stats — fair scoring: exclude uncertain zone (30-65%)
  const scorable = history.filter(h => {
    if (!h.verified) return false;
    const vr = _verificationResult(h);
    return vr.cls !== 'uncertain';
  });
  const correct = scorable.filter(h => h.correct).length;
  const accuracy = scorable.length > 0 ? ((correct / scorable.length) * 100).toFixed(0) : '—';
  const pending = history.filter(h => !h.verified).length;

  // Day-level accuracy: only count scorable predictions
  const dayBuckets = {};
  for (const h of scorable) {
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
        </div>
      </div>

      <div class="verdict" style="color:${color}">
        ${_verdictText(latest)}
        <span class="verdict-conf ${confColor}">Confiança ${latest.confidence.toLowerCase()}${trend.arrow ? ` · prob. ${trend.label} ${trend.arrow}` : ''}</span>
      </div>

      ${_mlCorrectionSummary(latest)}

      ${renderDrivers(latest)}

      <div class="gate-indicator ${gateOpen ? 'open' : 'closed'}">
        <span class="gate-dot"></span>
        ${gateOpen
          ? `${signals.map(s => '<span class="gate-chip">' + s + '</span>').join('')}`
          : 'Sense senyals de pluja'
        }
      </div>

      <div class="prediction-meta">
        Encerts: <strong style="color:var(--accent-green)">${dayAccuracy}%</strong> per dia (${daysCorrect}/${daysTotal})
        · <strong>${accuracy}%</strong> per predicció (${correct}/${scorable.length})${pending > 0 ? ` · <span style="color:var(--text-muted)">${pending} pendents de verificar</span>` : ''}
      </div>

      ${renderTechExpandable(latest)}
    </div>

    <!-- Probability chart -->
    <div class="chart-card">
      <h2>📈 Com ha canviat la probabilitat (últimes 24h)</h2>
      <div class="chart-card-body">
        <p class="chart-hint">Toca o passa el ratolí per veure els valors</p>
        <div class="chart-container">
          <canvas id="probChart"></canvas>
        </div>
      </div>
    </div>

    <div class="grid">
      <!-- Conditions -->
      <div class="info-card">
        <h3>🌡️ Condicions actuals</h3>
        <div class="info-card-body">${renderConditions(latest)}</div>
      </div>

      <!-- Radar -->
      <div class="info-card">
        <h3>📡 Radar</h3>
        <div class="info-card-body">${renderRadar(latest)}</div>
      </div>

      <!-- Atmospheric -->
      <div class="info-card">
        <h3>🌀 Atmosfera</h3>
        <div class="info-card-body">${renderAtmosphere(latest)}</div>
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
      <div class="history-card-body">
        <details class="pred-legend">
          <summary class="pred-legend-title">Com llegir les prediccions</summary>
          <div class="pred-legend-grid">
            <div class="pred-legend-section">
              <span class="pred-legend-heading">Vam dir</span>
              <span>☀️ <strong>No plourà</strong>, probabilitat &lt; 30%</span>
              <span>🌤️ <strong>Incert</strong>, entre 30% i 65%</span>
              <span>🌧️ <strong>Pluja probable</strong>, probabilitat ≥ 65%</span>
            </div>
            <div class="pred-legend-section">
              <span class="pred-legend-heading">Resultat</span>
              <span>✅ <strong>Encert</strong>, predicció segura correcta</span>
              <span>🔸 <strong>Encert/Error</strong>, predicció incerta (no compta al percentatge d'encerts)</span>
              <span>❌ <strong>Error</strong>, predicció segura incorrecta</span>
              <span>⏳ <strong>Pendent</strong>, encara no verificada</span>
            </div>
          </div>
        </details>
        <div id="calendar-root"></div>
      </div>
    </div>
  `;

  initCalendar(history);
  drawChart(history, latest);
}

function _cloudLayers(fv) {
  const lo = fv.cloud_cover_low, mi = fv.cloud_cover_mid, hi = fv.cloud_cover_high;
  if (lo == null && mi == null && hi == null) return '';
  const parts = [];
  if (lo != null) parts.push(`baix ${Math.round(lo)}%`);
  if (mi != null) parts.push(`mig ${Math.round(mi)}%`);
  if (hi != null) parts.push(`alt ${Math.round(hi)}%`);
  return ` <span style="font-size:11px;color:var(--text-muted)">(${parts.join(', ')})</span>`;
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
  const cloud = fv.cloud_cover;
  const gusts = fv.wind_gusts_10m;
  return `
    <div class="stat-row"><span class="stat-label">Temperatura</span><span class="stat-value">${c.temperature || '—'}°C</span></div>
    <div class="stat-row"><span class="stat-label">Humitat</span><span class="stat-value">${c.humidity || '—'}%</span></div>
    <div class="stat-row"><span class="stat-label">Punt de rosada</span><span class="stat-value">${dewPoint}</span></div>
    <div class="stat-row"><span class="stat-label">Pressió</span><span class="stat-value" style="${pressColor}">${c.pressure || '—'} hPa${pressTrend}</span></div>
    <div class="stat-row"><span class="stat-label">Vent</span><span class="stat-value">${c.wind_speed || '—'} km/h ${c.wind_dir || ''}${gusts != null ? ' (ràfegues ' + Math.round(gusts) + ')' : ''}</span></div>
    <div class="stat-row"><span class="stat-label">Cel cobert</span><span class="stat-value">${cloud != null ? Math.round(cloud) + '%' : '—'}${_cloudLayers(fv)}</span></div>
    <div class="stat-row"><span class="stat-label">Radiació solar</span><span class="stat-value">${solar != null ? Math.round(solar) + ' W/m² ' + solarDesc : '—'}</span></div>
    <div class="stat-row"><span class="stat-label">Pluja avui</span><span class="stat-value">${c.rain_today || '0.0'} mm</span></div>
  `;
}

function renderRadar(d) {
  const r = d.radar || {};
  const fv = d.feature_vector || {};
  const lightning = fv.lightning_count_30km;
  const lightningText = lightning != null ? (lightning > 0 ? `⚡ ${Math.round(lightning)} detectats` : 'Cap activitat') : '—';

  const view = deriveRadarViewModel(d);

  // Source agreement/disagreement note
  let sourceNote = '';
  if (!view.rvHasEcho && view.aemetHasEcho) {
    sourceNote = `<div class="stat-row"><span class="stat-label">Font</span><span class="stat-value" style="color:var(--accent-blue)">Radar AEMET (a ${view.aemetDist != null ? view.aemetDist + ' km' : '?'}, ${view.aemetCovPct != null ? view.aemetCovPct + '%' : '?'} cob.)</span></div>`;
  } else if (view.rvHasEcho && view.aemetHasEcho) {
    sourceNote = `<div class="stat-row"><span class="stat-label">Radar AEMET</span><span class="stat-value" style="color:var(--accent-blue)">Confirma pluja</span></div>`;
  } else if (view.rvHasEcho && !view.aemetHasEcho && fv.aemet_radar_has_echo != null) {
    sourceNote = `<div class="stat-row"><span class="stat-label">Radar AEMET</span><span class="stat-value" style="color:var(--text-muted)">No confirma</span></div>`;
  }

  // Storm movement direction text
  const vns = fv.radar_storm_velocity_ns;
  const vew = fv.radar_storm_velocity_ew;
  const vel = r.storm_velocity_kmh || fv.radar_storm_velocity_kmh || 0;
  let movementText = '';
  if (vel > 2 && vns != null && vew != null) {
    const dirs = [];
    if (Math.abs(vns) > 1) dirs.push(vns > 0 ? 'S' : 'N');
    if (Math.abs(vew) > 1) dirs.push(vew > 0 ? 'E' : 'W');
    if (dirs.length > 0) movementText = `→ ${dirs.join('')} a ${Math.round(vel)} km/h`;
    else movementText = `${Math.round(vel)} km/h`;
  }

  // Approaching text
  const approachingFlag = r.approaching || fv.radar_storm_approaching;
  let approachText = approachingFlag ? '⚠️ Sí' : 'No';
  if (movementText) approachText += ` (${movementText})`;
  if (r.storm_eta_min) approachText += ` ~${r.storm_eta_min} min`;

  // Direction text: use quadrant compass if available, else AEMET summary
  return `
    <div class="stat-row"><span class="stat-label">Pluja més propera</span><span class="stat-value">${view.nearestText}</span></div>
    <div class="stat-row"><span class="stat-label">Zona amb pluja</span><span class="stat-value">${view.coverageText}</span></div>
    <div class="stat-row"><span class="stat-label">S'acosta?</span><span class="stat-value">${approachText}</span></div>
    <div class="stat-row"><span class="stat-label">Direcció</span><span class="stat-value">${view.directionText}</span></div>
    <div class="stat-row"><span class="stat-label">Intensitat</span><span class="stat-value">${view.intensityText}</span></div>
    <div class="stat-row"><span class="stat-label">Llamps (30 km)</span><span class="stat-value">${lightningText}</span></div>
    ${sourceNote}
    <p class="card-hint">RainViewer (global) + AEMET (nacional), dos radars independents cada 10 min</p>
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

function _mlCorrectionSummary(d) {
  return ''; // Correction detail moved to technical section at bottom
}

function _renderBiasInsight(d) {
  const b = d.bias || {};
  if (b.temp == null && b.humidity == null) return '';
  const parts = [];
  if (b.temp != null && Math.abs(b.temp) >= 0.5) {
    parts.push(`${Math.abs(b.temp).toFixed(1)}°C més ${b.temp < 0 ? 'fred' : 'calent'} del previst`);
  }
  if (b.humidity != null && Math.abs(b.humidity) >= 3) {
    parts.push(`${Math.abs(b.humidity).toFixed(0)}% ${b.humidity > 0 ? 'més humit' : 'més sec'} del previst`);
  }
  if (parts.length === 0) return '';
  return `<p class="tech-explainer" style="margin-top:6px;font-style:italic">Ara mateix: ${parts.join(' i ')}, el sistema corregeix aquesta diferència.</p>`;
}

function renderDrivers(d) {
  const drivers = d.top_drivers;
  if (!drivers || drivers.length === 0) return '';

  const featureDrivers = drivers.filter(dr => dr.group !== 'Base (climatologia)');
  if (featureDrivers.length === 0) return '';

  const lines = selectDriverExplanations(d);
  if (lines.length === 0) return '';

  const naturalLines = lines.map(l => {
    const tip = GROUP_TOOLTIP[l.group] || '';
    const infoBtn = tip
      ? ` <span class="driver-info" tabindex="0" role="button" aria-label="Més info"><span class="driver-info-icon">ⓘ</span><span class="driver-info-tip">${tip}</span></span>`
      : '';
    return `<li class="driver-reason ${l.direction === 'pluja' ? 'rain' : 'dry'}">${l.icon} ${l.text}${infoBtn}</li>`;
  });

  return `
    <div class="drivers-section">
      <div class="drivers-title">Per què ${d.probability_pct}%?</div>
      <ul class="drivers-natural">
        ${naturalLines.join('')}
      </ul>
    </div>`;
}

function renderDriversTech(d) {
  const drivers = d.top_drivers;
  if (!drivers || drivers.length === 0) return '';

  const featureDrivers = drivers.filter(dr => dr.group !== 'Base (climatologia)');
  const bias = drivers.find(dr => dr.group === 'Base (climatologia)');
  if (featureDrivers.length === 0) return '';

  const maxAbs = Math.max(...featureDrivers.map(dr => Math.abs(dr.contribution)), 0.1);

  const techRows = featureDrivers.map(dr => {
    const pct = Math.min(Math.abs(dr.contribution) / maxAbs * 50, 50);
    const isRain = dr.direction === 'pluja';
    // Diverging bar: rain grows right from center, dry grows left from center
    const barStyle = isRain
      ? `left:50%;width:${pct.toFixed(1)}%`
      : `right:50%;width:${pct.toFixed(1)}%`;
    const barCls = isRain ? 'driver-bar-rain' : 'driver-bar-dry';
    const tip = GROUP_TOOLTIP[dr.group] || '';
    const infoBtn = tip
      ? ` <span class="driver-info" tabindex="0" role="button" aria-label="Més info"><span class="driver-info-icon">ⓘ</span><span class="driver-info-tip">${tip}</span></span>`
      : '';
    return `
      <div class="driver-row">
        <span class="driver-label">${dr.icon} ${dr.group}${infoBtn}</span>
        <div class="driver-bar-container">
          <div class="driver-bar-center"></div>
          <div class="driver-bar ${barCls}" style="${barStyle}"></div>
        </div>
      </div>`;
  }).join('');

  const baseText = bias
    ? `<div class="driver-base">🌟 Punt de partida: climatologia local de Cardedeu</div>`
    : '';

  return `
    <div class="drivers-tech-section">
      <div class="drivers-tech-intro">Pes de cada factor en la predicció actual</div>
      <div class="drivers-tech-legend">
        <span class="legend-dry">☀️ Temps sec</span>
        <span class="legend-rain">🌧️ Pluja</span>
      </div>
      ${techRows}
      ${baseText}
    </div>`;
}

function renderTechExpandable(d) {
  const r = d.radar || {};
  const e = d.ensemble || {};
  const a = d.aemet || {};
  const fv = d.feature_vector || {};
  const pct = d.probability_pct;

  // Source checks — inputs our model analyzed, not independent validators
  const checks = [];

  const radarView = deriveRadarViewModel(d);
  checks.push({
    name: 'Radar (RainViewer)',
    confirms: radarView.radarVoteRain,
    detail: radarView.radarVoteDetail
  });

  const modelsRain = e.models_rain || 0;
  const totalModels = e.total_models || 4;
  const mn = fv.ensemble_min_precip, mx = fv.ensemble_max_precip;
  const rangeText = mn != null && mx != null && modelsRain > 0 ? ` (${mn.toFixed(1)}–${mx.toFixed(1)} mm)` : '';
  checks.push({
    name: `Models globals (${modelsRain}/${totalModels})`,
    confirms: modelsRain > totalModels / 2,
    detail: modelsRain === 0 ? 'Cap preveu pluja' : `${modelsRain} de ${totalModels} preveuen pluja${rangeText}`
  });

  const aemetProb = a.prob_precip;
  if (aemetProb != null) {
    checks.push({
      name: 'AEMET',
      confirms: aemetProb >= 40,
      detail: `${aemetProb}% probabilitat de pluja`
    });
  }

  const lightning = fv.lightning_count_30km;
  if (lightning != null && lightning > 0) {
    checks.push({ name: 'Llamps', confirms: true, detail: `${Math.round(lightning)} detectats en 30 km` });
  }

  const stormNote = (a.prob_storm || 0) >= 10
    ? `<div class="source-vote storm-note"><span class="vote-badge rain">⚡</span><div class="vote-info"><span class="vote-name">Risc de tronada</span><span class="vote-detail">${a.prob_storm}%</span></div></div>`
    : '';

  const confirming = checks.filter(c => c.confirms).length;

  // Compact inline summary of each source
  const checkChips = checks.map(c => {
    const icon = c.confirms ? '🌧️' : '☀️';
    return `<span class="source-chip">${icon} ${c.name}: ${c.detail}</span>`;
  });

  const stormChip = (a.prob_storm || 0) >= 10
    ? `<span class="source-chip">⚡ Tronada: ${a.prob_storm}%</span>`
    : '';

  // Disagreement insight — only when our model contradicts the external sources
  const mlCat = d.rain_category;
  const mlRain = mlCat === 'probable' || (mlCat == null && pct >= 65);
  let disagreementNote = '';
  if (confirming > checks.length / 2 && !mlRain) {
    disagreementNote = `<p class="tech-explainer" style="margin-top:6px;font-style:italic">Diverses fonts indiquen pluja, però l'historial de Cardedeu mostra que aquest patró sovint és fals avís.</p>`;
  } else if (confirming <= checks.length / 2 && mlRain) {
    disagreementNote = `<p class="tech-explainer" style="margin-top:6px;font-style:italic">Les fonts convencionals no veuen pluja, però detectem patrons que històricament sí porten pluja aquí.</p>`;
  }

  const detailId = 'tech-detail-' + Date.now();
  return `
    <div class="tech-open-section">
      <p class="tech-explainer">
        Integrem ${d.features_used || '209'} variables en un model entrenat amb l'històric verificat de Cardedeu. Es re-entrena cada dia.
      </p>
      ${_renderBiasInsight(d)}
      <div class="sources-inline">
        <div class="sources-inline-label">Fonts:</div>
        <div class="sources-chips">
          ${checkChips.join('')}
          ${stormChip}
        </div>
      </div>
      ${disagreementNote}
      <button class="expand-toggle mini" onclick="this.classList.toggle('open');document.getElementById('${detailId}').classList.toggle('open')">
        <span class="chevron">▶</span> Detall tècnic
      </button>
      <div id="${detailId}" class="expand-content">
        ${renderDriversTech(d)}
      </div>
    </div>
  `;
}

function renderAtmosphere(d) {
  const p = d.pressure_levels || {};
  const w = d.wind_regime || {};
  const regimes = {
    llevantada: { icon: '🌊', name: 'Llevantada', desc: 'Humitat del mar contra les muntanyes, pluja #1 a Cardedeu (15% de probabilitat)', range: '60°-150° (E/SE)' },
    tramuntana: { icon: '❄️', name: 'Tramuntana', desc: 'Vent fred del nord/nord-est, supressor de pluja (5%)', range: '340°-60° (N/NE)' },
    migjorn: { icon: '🌡️', name: 'Migjorn', desc: 'Aire càlid africà, segon en pluja (15%)', range: '150°-190° (S)' },
    garbi: { icon: '🌀', name: 'Garbí', desc: 'Aire inestable del sud-oest, tempestes (11%)', range: '190°-250° (SW)' },
    ponent: { icon: '🏔️', name: 'Ponent', desc: 'Aire sec continental, supressor de pluja (6%)', range: '250°-340° (W/NW)' },
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
    // Fair scoring: only count sec/probable predictions, not uncertain zone
    const scorable = verified.filter(p => {
      const vr = _verificationResult(p);
      return vr.cls !== 'uncertain';
    });
    const correct = scorable.filter(p => p.correct).length;
    const anyRain = preds.some(p => p.actual_rain === true);
    const allPending = verified.length === 0;
    const hasPending = preds.some(p => !p.verified);
    const rainMm = preds.reduce((max, p) => Math.max(max, p.actual_rain_mm || 0), 0);
    const acc = scorable.length > 0 ? (correct / scorable.length) : null;
    const todayStr = new Date().toISOString().slice(0, 10);
    const isOngoing = dayKey >= todayStr;
    return { preds, verified, scorable, correct, anyRain, allPending, hasPending, isOngoing, rainMm, acc, count: preds.length };
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

    // Month stats — fair scoring: only count sec/probable predictions
    let mVerified = 0, mCorrect = 0, mPreds = 0, mRainDays = 0, mDaysCorrect = 0, mDaysWithData = 0;
    for (let d = 1; d <= daysInMonth; d++) {
      const key = `${year}-${String(month + 1).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
      const s = getDaySummary(key);
      if (s) {
        mPreds += s.count;
        mVerified += s.scorable.length;
        mCorrect += s.correct;
        if (s.anyRain) mRainDays++;
        if (s.scorable.length > 0) {
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
        ? `🌧️ En curs, ha plogut (${s.rainMm.toFixed(1)} mm)`
        : '📊 En curs, encara no ha plogut';
    } else if (s.anyRain) {
      resolutionCls = 'rain';
      resolutionText = `🌧️ Va ploure (${s.rainMm.toFixed(1)} mm)`;
    } else {
      resolutionCls = 'no-rain';
      resolutionText = '☀️ No va ploure';
    }

    let accCls = '', accText = '';
    if (s.scorable.length > 0) {
      const pct = ((s.correct / s.scorable.length) * 100).toFixed(0);
      accCls = pct >= 90 ? 'perfect' : pct >= 60 ? 'good' : 'bad';
      accText = `${s.correct}/${s.scorable.length} encerts`;
    }

    const predRows = s.preds.map(p => {
      const t = new Date(p.timestamp);
      const time = t.toLocaleString('ca-ES', { hour: '2-digit', minute: '2-digit' });
      const pct = p.probability_pct;
      const color = getProbColor(pct);
      const said = _predictionLabel(p);
      const vr = _verificationResult(p);
      const rainInfo = p.actual_rain_mm != null ? `${p.actual_rain_mm.toFixed(1)} mm` : '—';
      return `
        <div class="pred-row">
          <span class="pred-time">${time}</span>
          <div class="pred-prob-bar"><div class="pred-prob-fill" style="width:${pct}%;background:${color}"></div></div>
          <span class="pred-pct" style="color:${color}">${pct}%</span>
          <span class="pred-said">${said}</span>
          <span class="pred-result ${vr.cls}">${vr.text}</span>
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

  // Data — try 24h, widen to 48h / 7d if gap
  const now = Date.now();
  let points = [];
  for (const hours of [24, 48, 168]) {
    const cutoff = now - hours * 60 * 60 * 1000;
    points = history
      .filter(h => new Date(h.timestamp).getTime() > cutoff)
      .map(h => ({ t: new Date(h.timestamp).getTime(), p: h.probability_pct }));
    if (points.length >= 2) break;
  }

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

    // Uncertain zone (30%-65%) — shaded band
    const zoneTop = yScale(65);
    const zoneBot = yScale(30);
    ctx.fillStyle = 'rgba(210,153,34,0.08)';
    ctx.fillRect(pad.left, zoneTop, cW, zoneBot - zoneTop);

    // Zone boundary lines
    ctx.strokeStyle = 'rgba(210,153,34,0.25)';
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, zoneTop);
    ctx.lineTo(W - pad.right, zoneTop);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pad.left, zoneBot);
    ctx.lineTo(W - pad.right, zoneBot);
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

    // Zone labels
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(210,153,34,0.5)';
    ctx.fillText('incert', pad.left + 4, yScale(48) + 3);

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
let _lastHistory = [];

async function loadAndRender() {
  const latest = await fetchJSON('latest_prediction.json');
  let history = _lastHistory || [];
  try {
    history = await fetchJSONL('predictions_log.jsonl');
    _lastHistory = history;
  } catch (e) {
    console.warn('History load failed (using cached):', e.message);
  }
  const isUpdate = _latestTimestamp && _latestTimestamp !== latest.timestamp;
  _latestTimestamp = latest.timestamp;
  renderPrediction(latest, history);
  if (isUpdate) {
    const card = document.querySelector('.prediction-card');
    if (card) { card.classList.add('flash'); setTimeout(() => card.classList.remove('flash'), 1500); }
  }
  return { latest, history };
}

// Tooltip toggle: tap to open/close, tap outside to dismiss
// Uses position:fixed + JS positioning to avoid clipping by parent containers
function _positionTooltip(infoEl) {
  const tip = infoEl.querySelector('.driver-info-tip');
  if (!tip) return;
  // Temporarily show to measure
  tip.style.visibility = 'hidden';
  tip.style.display = 'block';
  const iconRect = infoEl.getBoundingClientRect();
  const tipRect = tip.getBoundingClientRect();
  tip.style.visibility = '';

  const margin = 12;
  // Horizontal: center on icon, clamp to viewport
  let left = iconRect.left + iconRect.width / 2 - tipRect.width / 2;
  left = Math.max(margin, Math.min(left, window.innerWidth - tipRect.width - margin));
  // Vertical: above icon, or below if no room above
  let top = iconRect.top - tipRect.height - 8;
  if (top < margin) top = iconRect.bottom + 8;

  tip.style.left = left + 'px';
  tip.style.top = top + 'px';

  // Position arrow to point at icon center
  const arrowLeft = iconRect.left + iconRect.width / 2 - left - 5;
  tip.style.setProperty('--arrow-left', arrowLeft + 'px');
}

function _clearTooltip(el) {
  el.classList.remove('active');
  const tip = el.querySelector('.driver-info-tip');
  if (tip) {
    tip.style.left = '';
    tip.style.top = '';
    tip.style.display = '';
    tip.style.visibility = '';
  }
}

function _handleTooltipClick(e) {
  const infoBtn = e.target.closest('.driver-info');
  document.querySelectorAll('.driver-info.active').forEach(el => {
    if (el !== infoBtn) _clearTooltip(el);
  });
  if (infoBtn) {
    e.preventDefault();
    e.stopPropagation();
    if (infoBtn.classList.contains('active')) {
      _clearTooltip(infoBtn);
    } else {
      infoBtn.classList.add('active');
      _positionTooltip(infoBtn);
    }
  }
}

document.addEventListener('click', _handleTooltipClick);
document.addEventListener('touchend', _handleTooltipClick);

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
