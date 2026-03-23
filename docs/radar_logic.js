const RADAR_MIN_DBZ = 10;

function _isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function _pctFromFraction(value) {
  if (!_isFiniteNumber(value)) {
    return null;
  }
  return Math.round(value * 100);
}

function _hasRainViewerEcho(radar) {
  const hasEcho = !!radar.has_echo;
  const nearest = radar.nearest_echo_km;
  const maxDbz = radar.max_dbz_20km || 0;
  return hasEcho || (_isFiniteNumber(nearest) && nearest < 30 && maxDbz >= RADAR_MIN_DBZ);
}

function _hasAemetEcho(featureVector) {
  const hasEchoCardedeu = (featureVector.aemet_radar_has_echo || 0) > 0;
  const hasEchoArea = (featureVector.aemet_radar_echoes_found || 0) > 0;
  return hasEchoCardedeu || hasEchoArea;
}

function _intensityLabel(dbz) {
  if (dbz >= 40) return "Forta";
  if (dbz >= 25) return "Moderada";
  return "Feble";
}

export function deriveRadarViewModel(data) {
  const radar = data.radar || {};
  const featureVector = data.feature_vector || {};

  const rvHasEcho = _hasRainViewerEcho(radar);
  const aemetHasEcho = _hasAemetEcho(featureVector);
  const bestHasEcho = rvHasEcho || aemetHasEcho;

  const aemetDbz = featureVector.aemet_radar_max_dbz_20km || 0;
  const aemetDist = featureVector.aemet_radar_nearest_echo_km;
  const aemetCovPct = _pctFromFraction(featureVector.aemet_radar_coverage_20km);

  const bestDbz = bestHasEcho
    ? (rvHasEcho ? (radar.max_dbz_20km || radar.dbz || 0) : aemetDbz)
    : 0;
  const bestDist = rvHasEcho ? radar.nearest_echo_km : aemetDist;
  const bestCov = bestHasEcho
    ? (rvHasEcho ? _pctFromFraction(radar.coverage_20km) : aemetCovPct)
    : 0;

  let nearestText = "No detectada";
  if (bestHasEcho && _isFiniteNumber(bestDist) && bestDist < 30) {
    nearestText = `${bestDist} km ${rvHasEcho ? (radar.nearest_echo_compass || "") : ""}`.trim();
  } else if (bestHasEcho && _isFiniteNumber(bestDist) && bestDist >= 30) {
    nearestText = `>${bestDist} km`;
  }

  const coverageText = `${bestCov != null ? bestCov : 0}% (radi 20 km)`;

  let intensityText = "Res detectat";
  if (bestHasEcho && bestDbz > 0) {
    intensityText = `${_intensityLabel(bestDbz)} (${Math.round(bestDbz)} dBZ)`;
  }

  const quadrants = radar.quadrants || {};
  const hasQuadrants = rvHasEcho && ["N", "E", "S", "W"].some((dir) => (quadrants[`max_dbz_${dir}`] || 0) > 5);

  let directionText = '<span style="color:var(--text-muted)">Sense pluja al radar</span>';
  if (hasQuadrants) {
    const compassParts = ["N", "E", "S", "W"].map((dir) => {
      const dbz = quadrants[`max_dbz_${dir}`] || 0;
      if (dbz > 5) return `<span style="color:var(--accent-blue);font-weight:700">${dir}</span>`;
      return `<span style="color:var(--text-muted);opacity:0.3">${dir}</span>`;
    }).join(" · ");
    directionText = compassParts;
  } else if (aemetHasEcho && _isFiniteNumber(aemetDist)) {
    directionText = `<span style="color:var(--accent-blue)">Pluja a ${aemetDist} km (AEMET)</span>`;
  }

  // Radar vote for "why prediction" section — derives from the same flags
  const approaching = !!(radar.approaching || featureVector.radar_storm_approaching);
  let radarVoteRain = bestHasEcho;
  let radarVoteDetail;
  if (rvHasEcho && aemetHasEcho) {
    radarVoteDetail = approaching
      ? `Pluja a ${bestDist || "?"} km, acostant-se (confirmat per 2 radars)`
      : "Pluja detectada per 2 radars independents";
  } else if (rvHasEcho) {
    radarVoteDetail = approaching
      ? `Pluja a ${bestDist || "?"} km, acostant-se`
      : `Pluja detectada a ${bestDist || "?"} km`;
  } else if (aemetHasEcho) {
    const intLabel = _intensityLabel(aemetDbz);
    radarVoteDetail = `Pluja ${intLabel.toLowerCase()} a ${aemetDist != null ? aemetDist + " km" : "?"}${aemetCovPct != null ? ", " + aemetCovPct + "% cobertura" : ""} (AEMET)`;
  } else {
    radarVoteDetail = "Sense pluja en 30 km";
  }

  // Gate signal text for the rain gate indicator chip
  let gateSignalText = null;
  if (bestHasEcho) {
    const distPart = _isFiniteNumber(bestDist) && bestDist < 30 ? ` (${bestDist} km)` : "";
    gateSignalText = `📡 Radar${distPart}`;
  }

  return {
    rvHasEcho,
    aemetHasEcho,
    bestHasEcho,
    bestDbz,
    bestDist,
    bestCov,
    nearestText,
    coverageText,
    intensityText,
    directionText,
    aemetDbz,
    aemetDist,
    aemetCovPct,
    approaching,
    radarVoteRain,
    radarVoteDetail,
    gateSignalText,
  };
}
