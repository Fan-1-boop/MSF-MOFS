"use strict";

/**
 * V6 hybrid satellite health algorithm.
 *
 * High-dimensional coupled subsystems (power / thermal / attitude):
 *   subsystem-local MSF-MOFS selected features -> logistic evidence -> stateful DBN.
 *
 * Auxiliary diagnostics and small subsystems continue to use the V5 parameter-level
 * evidence path. This preserves the existing API while allowing cross-channel
 * relation features to directly drive subsystem health.
 */

const fs = require("node:fs");
const path = require("node:path");
const v5 = require("./health-algorithm-v5.js");
const { extractFeatureVector } = require("./health-features.js");

const VERSION = "6.1.0-six-subsystem-local-health-dbn";
const MODEL_PATH = process.env.HEALTH_MSF_V6_MODEL
  ? path.resolve(process.env.HEALTH_MSF_V6_MODEL)
  : path.join(__dirname, "models", "subsystem-health-model-v6.json");

const TRANSITION = [
  [0.955, 0.043, 0.002],
  [0.075, 0.855, 0.070],
  [0.008, 0.092, 0.900],
];
const INITIAL_POSTERIOR = [0.965, 0.033, 0.002];
const localStateStore = new Map();
const evidenceStateStore = new Map();

const SUBSYSTEM_PARAMS = {
  power: ["param_1", "param_2", "param_8", "param_9"],
  thermal: ["param_3", "param_4", "param_9"],
  attitude: ["param_6", "param_7", "param_12"],
  communication: ["param_11"],
  data: ["param_10"],
  propulsion: ["param_5"],
};

function clamp(value, min = 0, max = 1) { return Math.min(max, Math.max(min, value)); }
function sigmoid(value) {
  if (value >= 0) { const z = Math.exp(-Math.min(60, value)); return 1 / (1 + z); }
  const z = Math.exp(Math.max(-60, value)); return z / (1 + z);
}
function logit(p) { const q = clamp(Number(p), 1e-6, 1 - 1e-6); return Math.log(q / (1 - q)); }
function normalPdf(value, center, sigma) { return Math.exp(-0.5 * ((value - center) / sigma) ** 2) + 1e-8; }

function loadLocalModel() {
  const raw = JSON.parse(fs.readFileSync(MODEL_PATH, "utf8"));
  if (!raw.subsystems || typeof raw.subsystems !== "object") throw new Error("V6 subsystem model is missing subsystems");
  return raw;
}
let LOCAL_MODEL = loadLocalModel();

function localKey(satelliteId, subsystemId) { return `${satelliteId || "default"}::${subsystemId}`; }
function updatePosterior(previous, evidence) {
  const predicted = [0, 0, 0];
  for (let from = 0; from < 3; from += 1) {
    for (let to = 0; to < 3; to += 1) predicted[to] += previous[from] * TRANSITION[from][to];
  }
  const likelihood = [normalPdf(evidence, 0.05, 0.18), normalPdf(evidence, 0.50, 0.23), normalPdf(evidence, 0.90, 0.18)];
  const unnormalized = predicted.map((value, index) => value * likelihood[index]);
  const total = unnormalized.reduce((sum, value) => sum + value, 0) || 1;
  return unnormalized.map((value) => value / total);
}
function smoothLocalEvidence(satelliteId, subsystemId, rawRisk, hardLimitViolation = false) {
  const key = localKey(satelliteId, subsystemId);
  if (hardLimitViolation) { evidenceStateStore.set(key, 1); return 1; }
  const previous = evidenceStateStore.has(key) ? Number(evidenceStateStore.get(key)) : 0.02;
  const alpha = subsystemId === "communication" ? 0.12 : 0.22;
  const smoothed = clamp((1 - alpha) * previous + alpha * clamp(rawRisk), 0, 1);
  evidenceStateStore.set(key, smoothed);
  return smoothed;
}

function updateLocalState(satelliteId, subsystemId, evidence) {
  const key = localKey(satelliteId, subsystemId);
  const previous = localStateStore.get(key) || INITIAL_POSTERIOR;
  const posterior = updatePosterior(previous, evidence);
  localStateStore.set(key, posterior);
  return posterior;
}
function scoreFromPosterior(p) { return Math.round(clamp(100 * (p[0] + 0.55 * p[1]), 0, 100)); }
function stateName(score) { return score >= 85 ? "健康" : score >= 60 ? "退化" : "故障"; }

function currentValueMap(telemetry) {
  const result = new Map();
  (Array.isArray(telemetry.coreParameters) ? telemetry.coreParameters : []).forEach((p) => {
    if (p?.key) result.set(p.key, Number(p.value ?? p.currentValue));
  });
  return result;
}

function hardViolation(telemetry, parameterKeys) {
  const values = currentValueMap(telemetry);
  const thresholds = telemetry.thresholds || {};
  for (const key of parameterKeys || []) {
    const value = values.get(key);
    const raw = thresholds[key] || {};
    const min = Number(raw.min ?? raw.lower);
    const max = Number(raw.max ?? raw.upper);
    if (!Number.isFinite(value)) continue;
    if ((Number.isFinite(min) && value < min) || (Number.isFinite(max) && value > max)) return true;
  }
  return false;
}

function localEvidence(subsystemId, model, values, readiness, telemetry) {
  const selected = Array.isArray(model.selectedFeatures) ? model.selectedFeatures : [];
  const e = model.logisticEvidence || {};
  const means = Array.isArray(e.mean) ? e.mean : [];
  const scales = Array.isArray(e.scale) ? e.scale : [];
  const coef = Array.isArray(e.coef) ? e.coef : [];
  let score = Number(e.intercept) || 0;
  const contributions = [];
  let available = 0;
  selected.forEach((feature, index) => {
    const raw = Number(values[feature.name]);
    const fallback = Number(model.imputationMedians?.[feature.name]);
    const value = Number.isFinite(raw) ? raw : (Number.isFinite(fallback) ? fallback : 0);
    if (Number.isFinite(raw)) available += 1;
    const scale = Math.abs(Number(scales[index])) > 1e-12 ? Number(scales[index]) : 1;
    const z = (value - (Number(means[index]) || 0)) / scale;
    const contribution = (Number(coef[index]) || 0) * z;
    score += contribution;
    contributions.push({ name: feature.name, featureType: feature.featureType, descriptor: feature.descriptor, scale: feature.scale, contribution });
  });
  const probability = sigmoid(score);
  const threshold = Number(e.threshold) || 0.5;
  const relativeLogit = score - logit(threshold);
  const calibrated = sigmoid(relativeLogit * (Number(e.riskLogitSlope) || 2.5));
  const ready = clamp(readiness, 0, 1);
  let risk = clamp(0.02 + ready * (calibrated - 0.02), 0, 0.82);
  const hard = hardViolation(telemetry, SUBSYSTEM_PARAMS[subsystemId]);
  if (hard) risk = 1;
  return {
    probability,
    risk,
    hardLimitViolation: hard,
    available,
    expected: selected.length,
    confidence: ready * (available / Math.max(1, selected.length)),
    contributions: contributions.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)),
  };
}

function unionSelectedNames() {
  const names = [];
  const seen = new Set();
  Object.values(LOCAL_MODEL.subsystems || {}).forEach((model) => {
    (model.selectedFeatures || []).forEach((feature) => {
      if (!seen.has(feature.name)) { seen.add(feature.name); names.push(feature.name); }
    });
  });
  return names;
}

function calculate(input) {
  const base = v5.calculate(input);
  const telemetry = input?.telemetry && typeof input.telemetry === "object" ? input.telemetry : input;
  const satelliteId = String(input?.satelliteId || telemetry?.satelliteId || "default");
  const names = unionSelectedNames();
  const vector = extractFeatureVector(telemetry, { selectedFeatureNames: names, scales: [8, 24, 64] });
  const localAssessments = {};

  Object.entries(LOCAL_MODEL.subsystems).forEach(([subsystemId, model]) => {
    const evidence = localEvidence(subsystemId, model, vector.features, vector.readiness, telemetry);
    const smoothedEvidence = smoothLocalEvidence(satelliteId, subsystemId, evidence.risk, evidence.hardLimitViolation);
    const posterior = updateLocalState(satelliteId, subsystemId, smoothedEvidence);
    const score = scoreFromPosterior(posterior);
    localAssessments[subsystemId] = {
      id: subsystemId,
      score,
      state: stateName(score),
      evidence: Number(smoothedEvidence.toFixed(6)),
      rawEvidence: Number(evidence.risk.toFixed(6)),
      evidenceProbability: Number(evidence.probability.toFixed(6)),
      hardLimitViolation: evidence.hardLimitViolation,
      confidence: Number(evidence.confidence.toFixed(4)),
      selectedFeatureCount: model.selectedFeatureCount,
      probabilities: {
        healthy: Number(posterior[0].toFixed(6)),
        degraded: Number(posterior[1].toFixed(6)),
        fault: Number(posterior[2].toFixed(6)),
      },
      contributors: evidence.contributions.slice(0, 8).map((item) => ({ ...item, contribution: Number(item.contribution.toFixed(5)) })),
      source: "subsystem-local-msf-v6",
    };
  });

  const subsystems = (base.subsystems || []).map((item) => localAssessments[item.id] ? { ...item, ...localAssessments[item.id], name: item.name } : item);
  const subsystemHealth = subsystems.map((item) => ({ name: item.name, value: item.score, percentage: item.score }));
  const criticality = new Map((v5.definitions?.subsystems || []).map((item) => [item.id, Number(item.criticality) || 0.5]));
  let missionSuccess = 1;
  let criticalityTotal = 0;
  let weightedSubsystemHealth = 0;
  const globalProbabilities = { healthy: 0, degraded: 0, fault: 0 };
  subsystems.forEach((item) => {
    const weight = criticality.get(item.id) || 0.5;
    const p = item.probabilities || { healthy: 1, degraded: 0, fault: 0 };
    const risk = Number(p.fault || 0) + 0.35 * Number(p.degraded || 0);
    missionSuccess *= 1 - weight * clamp(risk);
    criticalityTotal += weight;
    weightedSubsystemHealth += weight * Number(item.score || 0);
    globalProbabilities.healthy += weight * Number(p.healthy || 0);
    globalProbabilities.degraded += weight * Number(p.degraded || 0);
    globalProbabilities.fault += weight * Number(p.fault || 0);
  });
  const denom = criticalityTotal || 1;
  globalProbabilities.healthy /= denom;
  globalProbabilities.degraded /= denom;
  globalProbabilities.fault /= denom;
  const missionScore = 100 * clamp(missionSuccess);
  const weightedHealth = weightedSubsystemHealth / denom;
  const overallScore = Math.round(clamp(0.70 * weightedHealth + 0.30 * missionScore, 0, 100));

  return {
    ...base,
    algorithmVersion: VERSION,
    modelVersion: LOCAL_MODEL.schemaVersion,
    overallScore,
    overallState: stateName(overallScore),
    globalProbabilities: {
      healthy: Number(globalProbabilities.healthy.toFixed(6)),
      degraded: Number(globalProbabilities.degraded.toFixed(6)),
      fault: Number(globalProbabilities.fault.toFixed(6)),
    },
    globalFaultProbability: Number(globalProbabilities.fault.toFixed(6)),
    subsystemHealth,
    childHealthInfo: [subsystemHealth.map((item) => item.value), subsystemHealth.map((item) => 100 - item.value)],
    totalHealthInfo: [{ value: overallScore, name: "健康" }, { value: 100 - overallScore, name: "不健康" }],
    subsystems,
    msfV6: {
      mode: "subsystem-local",
      candidateFeatureCountGlobal: 582,
      selectedFeatureCountUnion: names.length,
      windowReadiness: Number(vector.readiness.toFixed(4)),
      subsystemModels: Object.fromEntries(Object.entries(LOCAL_MODEL.subsystems).map(([id, model]) => [id, {
        selectionMethod: model.selectionMethod,
        candidateFeatureCount: model.candidateFeatureCount,
        selectedFeatureCount: model.selectedFeatureCount,
        guidedCardinalities: model.guidedCardinalities,
        selectedSeed: model.selectedSeed,
      }])),
    },
  };
}

function reset(satelliteId) {
  v5.reset(satelliteId);
  const prefix = `${satelliteId || "default"}::`;
  for (const key of localStateStore.keys()) if (key.startsWith(prefix)) localStateStore.delete(key);
  for (const key of evidenceStateStore.keys()) if (key.startsWith(prefix)) evidenceStateStore.delete(key);
}

function snapshotState(satelliteId) {
  const id = String(satelliteId || "default");
  const prefix = `${id}::`;
  const localStates = {};
  const evidenceStates = {};
  for (const [key, value] of localStateStore.entries()) if (key.startsWith(prefix)) localStates[key.slice(prefix.length)] = value.map(Number);
  for (const [key, value] of evidenceStateStore.entries()) if (key.startsWith(prefix)) evidenceStates[key.slice(prefix.length)] = Number(value);
  return {
    schemaVersion: "health-dbn-state-v6-1.0",
    satelliteId: id,
    algorithmVersion: VERSION,
    modelSchemaVersion: LOCAL_MODEL.schemaVersion,
    updatedAt: new Date().toISOString(),
    base: v5.snapshotState(id),
    localStates,
    evidenceStates,
  };
}

function restoreState(satelliteId, snapshot) {
  const id = String(satelliteId || "default");
  reset(id);
  if (snapshot?.base) v5.restoreState(id, snapshot.base);
  let restored = 0;
  Object.entries(snapshot?.localStates || {}).forEach(([subsystemId, raw]) => {
    if (!Array.isArray(raw) || raw.length !== 3) return;
    const values = raw.map(Number); const total = values.reduce((a, b) => a + b, 0);
    if (!values.every((v) => Number.isFinite(v) && v >= 0) || !(total > 0)) return;
    localStateStore.set(localKey(id, subsystemId), values.map((v) => v / total)); restored += 1;
  });
  Object.entries(snapshot?.evidenceStates || {}).forEach(([subsystemId, raw]) => {
    const value = Number(raw); if (Number.isFinite(value)) evidenceStateStore.set(localKey(id, subsystemId), clamp(value));
  });
  return restored;
}

function reloadModel() {
  const base = v5.reloadModel();
  LOCAL_MODEL = loadLocalModel();
  localStateStore.clear();
  evidenceStateStore.clear();
  return { base, local: modelInfo() };
}

function modelInfo() {
  const counts = Object.fromEntries(Object.entries(LOCAL_MODEL.subsystems || {}).map(([id, m]) => [id, { candidate: m.candidateFeatureCount, selected: m.selectedFeatureCount }]));
  return {
    path: MODEL_PATH,
    schemaVersion: LOCAL_MODEL.schemaVersion,
    createdAt: LOCAL_MODEL.createdAt || null,
    method: "Subsystem-local hybrid feature evidence + logistic evidence + DBN",
    candidateFeatureCount: 582,
    selectedFeatureCount: unionSelectedNames().length,
    selectedFeatureRatio: Number((unionSelectedNames().length / 582).toFixed(6)),
    scales: [8, 24, 64],
    subsystemModels: counts,
    evidenceModel: "subsystem-local logistic",
    baseModel: v5.modelInfo(),
  };
}

function stateCount() { return v5.stateCount() + localStateStore.size + evidenceStateStore.size; }

module.exports = {
  VERSION,
  MODEL_PATH,
  calculate,
  reset,
  reloadModel,
  modelInfo,
  snapshotState,
  restoreState,
  stateCount,
  definitions: v5.definitions,
};
