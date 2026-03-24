// Driver explanation logic — extracted for testability
// Used by renderDrivers() in app.js

/**
 * Generate a human-readable explanation for a feature group.
 * Returns null when the explanation would be tautological, misleading, or vacuous.
 *
 * @param {string} group - Feature group name (e.g. 'Models globals', 'Radar')
 * @param {string} direction - 'pluja' or 'sec'
 * @param {object} ctx - Context with sensor values and ensemble data
 * @returns {string|null}
 */
export function explainGroup(group, direction, ctx) {
  const { rh, cloud, pressChange, solar, radarKm, radarCov, rainAccum,
          cape, li, lightning, ensemble } = ctx;
  const isRain = direction === 'pluja';

  switch (group) {
    case 'Models globals':
      if (isRain) return 'Les condicions meteorològiques afavoreixen pluja';
      return 'Les condicions generals no afavoreixen pluja';
    case 'Consistència NWP':
      return isRain ? 'La situació de pluja és persistent' : null;
    case 'Pluja confirmada':
      if (isRain) return rainAccum > 0 ? `Ja plou (${rainAccum.toFixed(1)} mm en 3h)` : null;
      return null;
    case 'Radar':
      if (isRain) {
        if (radarCov != null && radarCov > 0) return 'Detectem pluja a prop';
        if (radarKm != null && radarKm < 25) return `Detectem pluja a ${Math.round(radarKm)} km`;
        return null;
      }
      return null;
    case 'Humitat':
      if (rh == null) return isRain ? 'L\'aire és humit' : 'L\'aire és sec';
      return isRain ? `L'aire és humit (${Math.round(rh)}%)` : `L'aire és sec (${Math.round(rh)}%)`;
    case 'Aigua precipitable':
      return isRain ? 'Hi ha molta humitat a l\'atmosfera' : 'Poca humitat a l\'atmosfera';
    case 'Inestabilitat':
      if (isRain) {
        if ((cape != null && cape >= 300) || (li != null && li < 0)) return 'L\'atmosfera és inestable';
        return null;
      }
      return 'L\'atmosfera és estable';
    case 'Pressió':
      if (pressChange == null) return isRain ? 'La pressió baixa' : 'La pressió és estable';
      if (isRain) return pressChange <= 0 ? 'La pressió baixa' : null;
      return pressChange >= 0 ? 'La pressió puja o és estable' : null;
    case 'Règim de vent':
      return isRain ? 'El vent porta humitat del mar' : 'El vent no porta humitat';
    case 'Vent':
      return isRain ? 'El vent afavoreix pluja' : 'El vent no afavoreix pluja';
    case 'Núvols':
      if (cloud == null) return isRain ? 'Cel ennuvolat' : 'Cel obert';
      if (isRain) return cloud >= 50 ? `Cel ennuvolat (${Math.round(cloud)}%)` : null;
      return cloud < 50 ? `Cel obert (${Math.round(cloud)}% núvols)` : `Cel ennuvolat (${Math.round(cloud)}%), però no plourà`;
    case 'Temperatura':
      return isRain ? 'La temperatura afavoreix pluja' : null;
    case 'Hora del dia':
      return isRain ? 'Hora propensa a pluja' : null;
    case 'Radiació solar':
      if (solar == null) return isRain ? 'Poca llum solar' : 'Fa sol';
      if (isRain) return solar < 200 ? 'Cel cobert, poca llum solar' : null;
      return solar >= 200 ? 'Fa sol' : null;
    case 'Terra':
      return isRain ? 'El terra està humit' : null;
    case 'Capa límit':
      return isRain ? 'L\'aire es barreja i pot generar xàfecs' : null;
    case 'Llamps':
      if (isRain) {
        if (lightning != null && lightning > 0) return `Detectem ${Math.round(lightning)} llamps a prop`;
        return null;
      }
      return null;
    case 'Sentinella':
      return isRain ? 'Ja plou a localitats properes' : null;
    case 'Previsió oficial':
      return null;
    case 'Acord entre models':
      if (ensemble.models_rain != null) {
        const n = ensemble.models_rain, t = ensemble.total_models || 4;
        return isRain ? `${n} de ${t} fonts independents coincideixen` : null;
      }
      return isRain ? 'Diverses fonts independents coincideixen' : null;
    case 'Correcció local':
      return isRain ? 'L\'experiència local a Cardedeu ho confirma' : 'L\'experiència local a Cardedeu no hi dona suport';
    default:
      return null;
  }
}

/**
 * Brief tooltip explaining what each group measures, for general audience.
 * Shown on tap/hover of the info icon next to each driver explanation.
 */
export const GROUP_TOOLTIP = {
  'Models globals': 'Combinem 4 models meteorològics internacionals per veure si preveuen pluja a Cardedeu.',
  'Consistència NWP': 'Mirem si els models globals porten hores dient que plourà, o si canvien d\'opinió.',
  'Pluja confirmada': 'L\'estació meteorològica de Cardedeu mesura si ja està plovent.',
  'Radar': 'Dos radars (RainViewer i AEMET) escanegen si hi ha pluja en un radi de 30 km.',
  'Humitat': 'Mesurem quanta humitat hi ha a l\'aire. Més humitat = més probable que plogui.',
  'Aigua precipitable': 'Quanta aigua total hi ha acumulada a tota la columna d\'atmosfera.',
  'Inestabilitat': 'Si l\'atmosfera és inestable, l\'aire puja ràpid i pot generar tempestes.',
  'Pressió': 'Una pressió que baixa sol indicar que s\'acosta un front de pluja.',
  'Règim de vent': 'D\'on ve el vent a 1.500m d\'altura: del mar (humit) o de terra (sec).',
  'Vent': 'La velocitat i direcció del vent afecten si porta o s\'endú la humitat.',
  'Núvols': 'El percentatge de cel cobert per núvols, mesurat per satèl·lit.',
  'Temperatura': 'La temperatura afecta l\'evaporació i la formació de núvols.',
  'Hora del dia': 'Les tempestes d\'estiu solen ser de tarda; la pluja frontal pot ser a qualsevol hora.',
  'Radiació solar': 'Quanta llum del sol arriba a terra. Poca radiació = cel cobert.',
  'Terra': 'Si el terra ja està mullat, la pluja nova s\'escola més fàcilment i es detecta abans.',
  'Capa límit': 'La capa d\'aire més propera al terra. Si es barreja molt, pot generar xàfecs.',
  'Llamps': 'La xarxa de detecció de llamps (XDDE) mesura activitat elèctrica a prop.',
  'Sentinella': 'Estacions meteorològiques properes (Meteocat XEMA) detecten si ja plou als voltants.',
  'Acord entre models': 'Si diversos models globals coincideixen, la predicció és més fiable.',
  'Correcció local': 'Ajustem la predicció amb 12 anys d\'historial real de pluja a Cardedeu.',
};

/**
 * Return semantic concept tags for an explanation to prevent redundant lines.
 * Only groups that can overlap with others get tags; unique groups return [].
 */
const CONCEPT_MAP = {
  'Humitat|pluja': ['humidity'], 'Humitat|sec': ['dry_air'],
  'Aigua precipitable|pluja': ['humidity'], 'Aigua precipitable|sec': ['dry_air'],
  'Núvols|pluja': ['cloudy'], 'Núvols|sec': ['sunshine'],
  'Radiació solar|pluja': ['cloudy'], 'Radiació solar|sec': ['sunshine'],
  'Règim de vent|pluja': ['wind'], 'Règim de vent|sec': ['wind'],
  'Vent|pluja': ['wind'], 'Vent|sec': ['wind'],
  'Inestabilitat|sec': ['stable'],
  'Pluja confirmada|pluja': ['raining_nearby'],
  'Sentinella|pluja': ['raining_nearby'],
};

export function getConceptTags(group, direction, text) {
  return CONCEPT_MAP[`${group}|${direction}`] || [];
}

/** Collect up to maxCount non-null, non-redundant explanations from candidates. */
function _collectExplanations(candidates, direction, maxCount, ctx, usedConcepts) {
  const results = [];
  for (const dr of candidates) {
    if (results.length >= maxCount) break;
    const text = explainGroup(dr.group, direction, ctx);
    if (!text) continue;
    const concepts = getConceptTags(dr.group, direction, text);
    if (concepts.some(c => usedConcepts.has(c))) continue;
    concepts.forEach(c => usedConcepts.add(c));
    results.push({ icon: dr.icon, text, direction, group: dr.group });
  }
  return results;
}

/**
 * Select which driver explanations to show, based on probability and available data.
 * Deduplicates semantically overlapping explanations (e.g. two lines both saying "sec").
 *
 * @param {object} d - Full prediction data (top_drivers, feature_vector, ensemble, probability_pct)
 * @returns {Array<{icon: string, text: string, direction: string}>}
 */
export function selectDriverExplanations(d) {
  const drivers = d.top_drivers;
  if (!drivers || drivers.length === 0) return [];

  const featureDrivers = drivers.filter(dr => dr.group !== 'Base (climatologia)');
  if (featureDrivers.length === 0) return [];

  const rainPushers = featureDrivers.filter(dr => dr.direction === 'pluja').sort((a, b) => b.contribution - a.contribution);
  const dryPushers = featureDrivers.filter(dr => dr.direction === 'sec').sort((a, b) => a.contribution - b.contribution);

  const fv = d.feature_vector || {};
  const ctx = {
    cloud: fv.cloud_cover,
    rh: fv.relative_humidity_2m,
    pressChange: fv.pressure_change_3h,
    solar: fv.shortwave_radiation,
    radarKm: fv.radar_nearest_echo_km,
    radarCov: fv.radar_coverage_20km,
    rainAccum: fv.rain_accum_3h,
    cape: fv.cape,
    li: fv.nwp_lifted_index,
    lightning: fv.lightning_count_30km,
    ensemble: d.ensemble || {},
  };

  const pct = d.probability_pct;
  const isExtreme = pct < 10 || pct > 90;

  const topRain = rainPushers.slice(0, isExtreme ? 6 : 4);
  const topDry = dryPushers.slice(0, isExtreme ? 6 : 4);

  const usedConcepts = new Set();

  if (isExtreme && pct < 10) {
    return _collectExplanations(topDry, 'sec', 3, ctx, usedConcepts);
  } else if (isExtreme && pct > 90) {
    return _collectExplanations(topRain, 'pluja', 3, ctx, usedConcepts);
  } else {
    const results = _collectExplanations(topRain, 'pluja', 2, ctx, usedConcepts);
    results.push(..._collectExplanations(topDry, 'sec', 2, ctx, usedConcepts));
    return results;
  }
}
