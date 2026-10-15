"use strict";

/**
 * V6 MSF health feature builder.
 *
 * One sample = one multivariate telemetry window ending at the current time.
 * Candidate feature space:
 *   - 12 channels × 8 univariate descriptors × 3 scales = 288
 *   - 14 engineering relation pairs × 7 cross-channel descriptors × 3 scales = 294
 *   - total = 582 candidate features
 *
 * Cross-channel features are intentionally normalized by each channel's
 * engineering span. They represent relation/coupling changes that can be hard
 * to detect from the marginal distribution of any one channel.
 */

const DEFAULT_SCALES = [8, 24, 64];
const DESCRIPTORS = [
  "meanPosition",
  "stdRatio",
  "madRatio",
  "rangeRatio",
  "slopeRatio",
  "diffMaxRatio",
  "thresholdMargin",
  "violationRatio",
];

const RELATION_DESCRIPTORS = [
  "corr",
  "lag1Corr",
  "diffMean",
  "diffStd",
  "absDiffMean",
  "slopeDiff",
  "endpointDiff",
];

// Pairs are chosen from engineering subsystems. They are not claimed as a
// flight-qualified physical model; they form a controlled synthetic benchmark
// for coupled telemetry-health experiments.
const RELATION_PAIRS = [
  { id: "p01_p02", a: "param_1", b: "param_2", group: "power" },
  { id: "p01_p09", a: "param_1", b: "param_9", group: "power" },
  { id: "p01_p10", a: "param_1", b: "param_10", group: "power" },
  { id: "p02_p09", a: "param_2", b: "param_9", group: "power" },
  { id: "p02_p10", a: "param_2", b: "param_10", group: "power" },
  { id: "p09_p10", a: "param_9", b: "param_10", group: "power" },
  { id: "p03_p04", a: "param_3", b: "param_4", group: "thermal" },
  { id: "p03_p09", a: "param_3", b: "param_9", group: "thermal" },
  { id: "p04_p09", a: "param_4", b: "param_9", group: "thermal" },
  { id: "p06_p07", a: "param_6", b: "param_7", group: "attitude" },
  { id: "p06_p12", a: "param_6", b: "param_12", group: "attitude" },
  { id: "p07_p12", a: "param_7", b: "param_12", group: "attitude" },
  { id: "p10_p11", a: "param_10", b: "param_11", group: "communication" },
  { id: "p01_p08", a: "param_1", b: "param_8", group: "solar" },
];

function finite(value) {
  return value !== null && value !== "" && Number.isFinite(Number(value));
}

function mean(values) {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function median(values) {
  if (!values.length) return 0;
  const sorted = values.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function mad(values) {
  if (!values.length) return 0;
  const center = median(values);
  return median(values.map((value) => Math.abs(value - center)));
}

function std(values) {
  if (values.length < 2) return 0;
  const center = mean(values);
  const variance = values.reduce((sum, value) => sum + (value - center) ** 2, 0) / values.length;
  return Math.sqrt(Math.max(0, variance));
}

function regressionSlope(values) {
  if (values.length < 3) return 0;
  const xMean = (values.length - 1) / 2;
  const yMean = mean(values);
  let numerator = 0;
  let denominator = 0;
  for (let i = 0; i < values.length; i += 1) {
    numerator += (i - xMean) * (values[i] - yMean);
    denominator += (i - xMean) ** 2;
  }
  return denominator ? numerator / denominator : 0;
}

function pearson(a, b) {
  const n = Math.min(a.length, b.length);
  if (n < 3) return 0;
  const aa = a.slice(-n);
  const bb = b.slice(-n);
  const ma = mean(aa);
  const mb = mean(bb);
  let cov = 0;
  let va = 0;
  let vb = 0;
  for (let i = 0; i < n; i += 1) {
    const da = aa[i] - ma;
    const db = bb[i] - mb;
    cov += da * db;
    va += da * da;
    vb += db * db;
  }
  const denom = Math.sqrt(va * vb);
  return denom > 1e-12 ? Math.max(-1, Math.min(1, cov / denom)) : 0;
}

function resolveThreshold(parameter, thresholds = {}) {
  const raw = thresholds[parameter.key] || thresholds[parameter.name] || parameter.threshold || {};
  const minRaw = raw.min ?? raw.lower ?? parameter.minValue ?? parameter.min;
  const maxRaw = raw.max ?? raw.upper ?? parameter.maxValue ?? parameter.max;
  const min = finite(minRaw) ? Number(minRaw) : NaN;
  const max = finite(maxRaw) ? Number(maxRaw) : NaN;
  return { min, max };
}

function thresholdNormalizer(values, threshold) {
  const min = threshold.min;
  const max = threshold.max;
  const observedMin = Math.min(...values);
  const observedMax = Math.max(...values);
  const windowMean = mean(values);
  const thresholdSpan = Number.isFinite(min) && Number.isFinite(max) && max > min ? max - min : NaN;
  const fallbackSpan = Math.max(observedMax - observedMin, Math.abs(windowMean) * 0.25, 1e-6);
  const span = Number.isFinite(thresholdSpan) ? thresholdSpan : fallbackSpan;
  const center = Number.isFinite(min) && Number.isFinite(max) ? (min + max) / 2 : windowMean;
  return { min, max, span, center };
}

function signedMargin(value, min, max, span) {
  if (Number.isFinite(min) && value < min) return -(min - value) / span;
  if (Number.isFinite(max) && value > max) return -(value - max) / span;
  const margins = [];
  if (Number.isFinite(min)) margins.push((value - min) / span);
  if (Number.isFinite(max)) margins.push((max - value) / span);
  return margins.length ? Math.min(...margins) : 1;
}

function descriptorValues(values, threshold, requestedDescriptors = DESCRIPTORS) {
  const requested = requestedDescriptors instanceof Set
    ? requestedDescriptors
    : new Set(Array.isArray(requestedDescriptors) ? requestedDescriptors : DESCRIPTORS);
  const need = (name) => requested.has(name);
  const norm = thresholdNormalizer(values, threshold);
  const observedMin = Math.min(...values);
  const observedMax = Math.max(...values);
  const windowMean = mean(values);
  const out = {};

  if (need("meanPosition")) out.meanPosition = (windowMean - norm.center) / norm.span;
  if (need("stdRatio")) out.stdRatio = std(values) / norm.span;
  if (need("madRatio")) out.madRatio = (1.4826 * mad(values)) / norm.span;
  if (need("rangeRatio")) out.rangeRatio = (observedMax - observedMin) / norm.span;
  if (need("slopeRatio")) out.slopeRatio = regressionSlope(values) / norm.span;
  if (need("diffMaxRatio")) {
    let diffMax = 0;
    for (let i = 1; i < values.length; i += 1) diffMax = Math.max(diffMax, Math.abs(values[i] - values[i - 1]));
    out.diffMaxRatio = diffMax / norm.span;
  }
  if (need("thresholdMargin")) out.thresholdMargin = signedMargin(windowMean, norm.min, norm.max, norm.span);
  if (need("violationRatio")) {
    let violations = 0;
    for (const value of values) {
      if ((Number.isFinite(norm.min) && value < norm.min) || (Number.isFinite(norm.max) && value > norm.max)) violations += 1;
    }
    out.violationRatio = violations / values.length;
  }
  return out;
}

function normalizedSeries(values, threshold) {
  const norm = thresholdNormalizer(values, threshold);
  return values.map((value) => (value - norm.center) / norm.span);
}

function relationDescriptorValues(valuesA, thresholdA, valuesB, thresholdB, requestedDescriptors = RELATION_DESCRIPTORS) {
  const requested = requestedDescriptors instanceof Set
    ? requestedDescriptors
    : new Set(Array.isArray(requestedDescriptors) ? requestedDescriptors : RELATION_DESCRIPTORS);
  const need = (name) => requested.has(name);
  const n = Math.min(valuesA.length, valuesB.length);
  if (n < 2) return {};
  const a = normalizedSeries(valuesA.slice(-n), thresholdA);
  const b = normalizedSeries(valuesB.slice(-n), thresholdB);
  const diff = a.map((value, i) => value - b[i]);
  const out = {};
  if (need("corr")) out.corr = pearson(a, b);
  if (need("lag1Corr")) out.lag1Corr = n >= 4 ? pearson(a.slice(1), b.slice(0, -1)) : 0;
  if (need("diffMean")) out.diffMean = mean(diff);
  if (need("diffStd")) out.diffStd = std(diff);
  if (need("absDiffMean")) out.absDiffMean = mean(diff.map(Math.abs));
  if (need("slopeDiff")) out.slopeDiff = regressionSlope(a) - regressionSlope(b);
  if (need("endpointDiff")) out.endpointDiff = a[a.length - 1] - b[b.length - 1];
  return out;
}

function buildSeriesMap(telemetry) {
  const trendCharts = Array.isArray(telemetry.trendCharts) ? telemetry.trendCharts : [];
  const coreParameters = Array.isArray(telemetry.coreParameters) ? telemetry.coreParameters : [];
  const seriesMap = new Map();
  trendCharts.forEach((chart) => {
    const key = chart && (chart.key || chart.name);
    if (!key) return;
    const series = Array.isArray(chart.seriesData) ? chart.seriesData : Array.isArray(chart.data) ? chart.data : [];
    seriesMap.set(key, series.filter(finite).map(Number));
  });
  coreParameters.forEach((parameter) => {
    if (!parameter) return;
    if (!seriesMap.has(parameter.key) && finite(parameter.value ?? parameter.currentValue)) {
      seriesMap.set(parameter.key, [Number(parameter.value ?? parameter.currentValue)]);
    }
  });
  return seriesMap;
}

function buildSelectionPlan(selectedFeatureNames) {
  if (!Array.isArray(selectedFeatureNames) || !selectedFeatureNames.length) return null;
  const univariate = new Map();
  const relations = new Map();
  selectedFeatureNames.forEach((name) => {
    const text = String(name);
    let match = /^(param_\d+)__([^_]+)__w(\d+)$/.exec(text);
    if (match) {
      const [, parameterKey, descriptor, scaleRaw] = match;
      const scale = Number(scaleRaw);
      if (!DESCRIPTORS.includes(descriptor) || !Number.isFinite(scale)) return;
      if (!univariate.has(parameterKey)) univariate.set(parameterKey, new Map());
      const byScale = univariate.get(parameterKey);
      if (!byScale.has(scale)) byScale.set(scale, new Set());
      byScale.get(scale).add(descriptor);
      return;
    }
    match = /^rel_(param_\d+)_(param_\d+)__([^_]+)__w(\d+)$/.exec(text);
    if (match) {
      const [, a, b, descriptor, scaleRaw] = match;
      const scale = Number(scaleRaw);
      if (!RELATION_DESCRIPTORS.includes(descriptor) || !Number.isFinite(scale)) return;
      const pairKey = `${a}|${b}`;
      if (!relations.has(pairKey)) relations.set(pairKey, new Map());
      const byScale = relations.get(pairKey);
      if (!byScale.has(scale)) byScale.set(scale, new Set());
      byScale.get(scale).add(descriptor);
    }
  });
  return (univariate.size || relations.size) ? { univariate, relations } : null;
}

function extractFeatureVector(telemetry, options = {}) {
  const scales = Array.isArray(options.scales) && options.scales.length ? options.scales : DEFAULT_SCALES;
  const selectionPlan = buildSelectionPlan(options.selectedFeatureNames);
  const coreParameters = Array.isArray(telemetry.coreParameters) ? telemetry.coreParameters : [];
  const thresholds = telemetry.thresholds || {};
  const seriesMap = buildSeriesMap(telemetry);
  const parameterMap = new Map();
  coreParameters.forEach((parameter, index) => {
    const key = parameter.key || `param_${index + 1}`;
    parameterMap.set(key, { ...parameter, key });
  });

  const features = {};
  const metadata = {};
  let minimumHistory = Infinity;

  coreParameters.forEach((parameter, index) => {
    const key = parameter.key || `param_${index + 1}`;
    const series = seriesMap.get(key) || seriesMap.get(parameter.name) || [];
    minimumHistory = Math.min(minimumHistory, series.length);
    const threshold = resolveThreshold({ ...parameter, key }, thresholds);
    const plannedScales = selectionPlan?.univariate.get(key);
    const scalesForParameter = plannedScales ? Array.from(plannedScales.keys()) : (selectionPlan ? [] : scales);
    scalesForParameter.forEach((scale) => {
      const slice = series.slice(-scale).filter(Number.isFinite);
      if (slice.length < scale) return;
      const requestedDescriptors = plannedScales?.get(scale) || new Set(DESCRIPTORS);
      const values = descriptorValues(slice, threshold, requestedDescriptors);
      requestedDescriptors.forEach((descriptor) => {
        const name = `${key}__${descriptor}__w${scale}`;
        features[name] = values[descriptor];
        metadata[name] = {
          featureType: "univariate",
          parameterKey: key,
          parameterName: parameter.name || key,
          descriptor,
          scale,
          unit: parameter.unit || "",
        };
      });
    });
  });

  RELATION_PAIRS.forEach((pair) => {
    const pairKey = `${pair.a}|${pair.b}`;
    const plannedScales = selectionPlan?.relations.get(pairKey);
    const scalesForPair = plannedScales ? Array.from(plannedScales.keys()) : (selectionPlan ? [] : scales);
    if (!scalesForPair.length) return;
    const parameterA = parameterMap.get(pair.a) || { key: pair.a, name: pair.a };
    const parameterB = parameterMap.get(pair.b) || { key: pair.b, name: pair.b };
    const seriesA = seriesMap.get(pair.a) || [];
    const seriesB = seriesMap.get(pair.b) || [];
    const thresholdA = resolveThreshold(parameterA, thresholds);
    const thresholdB = resolveThreshold(parameterB, thresholds);
    scalesForPair.forEach((scale) => {
      const a = seriesA.slice(-scale).filter(Number.isFinite);
      const b = seriesB.slice(-scale).filter(Number.isFinite);
      if (a.length < scale || b.length < scale) return;
      const requestedDescriptors = plannedScales?.get(scale) || new Set(RELATION_DESCRIPTORS);
      const values = relationDescriptorValues(a, thresholdA, b, thresholdB, requestedDescriptors);
      requestedDescriptors.forEach((descriptor) => {
        const name = `rel_${pair.a}_${pair.b}__${descriptor}__w${scale}`;
        features[name] = values[descriptor];
        metadata[name] = {
          featureType: "relation",
          relationId: pair.id,
          relationGroup: pair.group,
          parameterKeys: [pair.a, pair.b],
          descriptor,
          scale,
        };
      });
    });
  });

  if (!Number.isFinite(minimumHistory)) minimumHistory = 0;
  const maxScale = Math.max(...scales);
  return {
    features,
    metadata,
    featureNames: Object.keys(features),
    minimumHistory,
    readiness: Math.min(1, minimumHistory / maxScale),
  };
}

function expectedFeatureNames(parameterKeys, scales = DEFAULT_SCALES, options = {}) {
  const result = [];
  parameterKeys.forEach((key) => {
    scales.forEach((scale) => {
      DESCRIPTORS.forEach((descriptor) => result.push(`${key}__${descriptor}__w${scale}`));
    });
  });
  if (options.includeRelations !== false) {
    RELATION_PAIRS.forEach((pair) => {
      scales.forEach((scale) => {
        RELATION_DESCRIPTORS.forEach((descriptor) => {
          result.push(`rel_${pair.a}_${pair.b}__${descriptor}__w${scale}`);
        });
      });
    });
  }
  return result;
}

module.exports = {
  DEFAULT_SCALES,
  DESCRIPTORS,
  RELATION_DESCRIPTORS,
  RELATION_PAIRS,
  extractFeatureVector,
  expectedFeatureNames,
  descriptorValues,
  relationDescriptorValues,
  buildSelectionPlan,
};
