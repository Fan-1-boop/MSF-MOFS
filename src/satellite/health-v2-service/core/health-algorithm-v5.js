"use strict";

/**
 * MSF-MOFS + Bayesian satellite health algorithm.
 *
 * Offline:
 *   telemetry windows -> 288 candidate features -> MSF-MOFS -> selected feature mask
 *   -> Gaussian class-conditional evidence model.
 *
 * Online:
 *   latest telemetry window -> selected features only -> Bayesian fault evidence
 *   -> stateful 3-state DBN per subsystem -> health degree.
 *
 * The search/training code is in ../../msf-training/train_msf_mofs.py.
 */

const fs = require("node:fs");
const path = require("node:path");
const { extractFeatureVector } = require("./health-features.js");

const VERSION = "4.0.0-msf-calibrated-dbn";
const MODEL_PATH = process.env.HEALTH_MSF_MODEL
  ? path.resolve(process.env.HEALTH_MSF_MODEL)
  : path.join(__dirname, "models", "health-model.json");

const SUBSYSTEMS = [
  { id: "power", name: "电源", criticality: 0.92 },
  { id: "attitude", name: "姿轨控", criticality: 0.92 },
  { id: "thermal", name: "热控", criticality: 0.58 },
  { id: "communication", name: "通信", criticality: 0.66 },
  { id: "data", name: "数据处理", criticality: 0.72 },
  { id: "propulsion", name: "推进", criticality: 0.78 },
];

const UNIT_DEFINITIONS = [
  { id: "thruster", key: "tlq", name: "推力器", parameters: ["param_5"], subsystems: { propulsion: 0.72 }, critical: true },
  { id: "sada", key: "sada", name: "SADA", parameters: ["param_8"], subsystems: { power: 0.12 }, critical: false },
  { id: "wheel", key: "dll", name: "动量轮", parameters: ["param_6"], subsystems: { attitude: 0.25 }, critical: true },
  { id: "gyro", key: "tly", name: "陀螺仪", parameters: ["param_7"], subsystems: { attitude: 0.32 }, critical: true },
  { id: "startracker", key: "xpb", name: "星敏感器", parameters: ["param_12"], subsystems: { attitude: 0.43 }, critical: true },
  { id: "powercontroller", key: "dykzq", name: "电源控制器", parameters: ["param_1", "param_9"], subsystems: { power: 0.34, thermal: 0.30 }, critical: true },
  { id: "computer", key: "xzjsj", name: "星载计算机", parameters: ["param_10"], subsystems: { data: 0.75 }, critical: true },
  { id: "transponder", key: "txj", name: "测控应答机", parameters: ["param_11"], subsystems: { communication: 0.72 }, critical: true },
  { id: "battery", key: "xdc", name: "蓄电池", parameters: ["param_2", "param_3"], subsystems: { power: 0.34, thermal: 0.28 }, critical: false },
  { id: "payload", key: "zaihe", name: "载荷", parameters: ["param_4"], subsystems: { thermal: 0.42 }, critical: true },
];

const PARAMETER_TO_SUBSYSTEMS = {};
UNIT_DEFINITIONS.forEach((unit) => {
  unit.parameters.forEach((parameterKey) => {
    PARAMETER_TO_SUBSYSTEMS[parameterKey] = {
      ...(PARAMETER_TO_SUBSYSTEMS[parameterKey] || {}),
      ...unit.subsystems,
    };
  });
});

const TRANSITION = [
  [0.955, 0.043, 0.002],
  [0.075, 0.855, 0.070],
  [0.008, 0.092, 0.900],
];
const INITIAL_POSTERIOR = [0.965, 0.033, 0.002];
const stateStore = new Map();

function clamp(value, min = 0, max = 1) {
  return Math.min(max, Math.max(min, value));
}

function loadModel() {
  try {
    const model = JSON.parse(fs.readFileSync(MODEL_PATH, "utf8"));
    if (!Array.isArray(model.selectedFeatures) || !model.selectedFeatures.length) {
      throw new Error("selectedFeatures is empty");
    }
    return model;
  } catch (error) {
    const wrapped = new Error(`无法加载 MSF 健康模型：${MODEL_PATH} (${error.message})`);
    wrapped.code = "MSF_MODEL_LOAD_FAILED";
    throw wrapped;
  }
}

let MODEL;
let SELECTED;
let FEATURE_BY_NAME;

function sigmoid(value) {
  if (value >= 0) {
    const z = Math.exp(-Math.min(value, 60));
    return 1 / (1 + z);
  }
  const z = Math.exp(Math.max(value, -60));
  return z / (1 + z);
}

let PARAMETER_EVIDENCE;
let EVIDENCE_CONFIG;
let RATIO_THRESHOLD;
let RISK_SLOPE;
let MODEL_RISK_CAP;
let TOP_WEIGHTS;

function applyModel(model) {
  MODEL = model;
  SELECTED = MODEL.selectedFeatures;
  FEATURE_BY_NAME = new Map(SELECTED.map((feature) => [feature.name, feature]));
  PARAMETER_EVIDENCE = MODEL.parameterEvidence || {};
  EVIDENCE_CONFIG = MODEL.evidenceAggregation || {};
  RATIO_THRESHOLD = Number(EVIDENCE_CONFIG.globalRatioThreshold) || 2.0;
  RISK_SLOPE = Number(EVIDENCE_CONFIG.riskSlope) || 2.5;
  MODEL_RISK_CAP = Number(EVIDENCE_CONFIG.modelRiskCap) || 0.75;
  TOP_WEIGHTS = Array.isArray(EVIDENCE_CONFIG.globalTopWeights)
    ? EVIDENCE_CONFIG.globalTopWeights.map(Number)
    : [0.65, 0.35];
}

function modelInfo() {
  return {
    path: MODEL_PATH,
    schemaVersion: MODEL.schemaVersion || null,
    createdAt: MODEL.createdAt || null,
    method: MODEL.method?.name || 'MSF-MOFS',
    candidateFeatureCount: MODEL.featureSpace?.candidateCount || 288,
    selectedFeatureCount: SELECTED.length,
    selectedFeatureRatio: Number((SELECTED.length / Math.max(1, MODEL.featureSpace?.candidateCount || 288)).toFixed(6)),
    scales: MODEL.featureSpace?.scales || [8, 24, 64],
    guidedCardinalities: MODEL.method?.guidedCardinalities || [],
    evidenceModel: EVIDENCE_CONFIG.model || 'parameter-local logistic',
  };
}

function reloadModel() {
  const model = loadModel();
  applyModel(model);
  // A new feature/evidence model must not continue with posteriors generated by the old model.
  stateStore.clear();
  return modelInfo();
}

applyModel(loadModel());

function featureValue(name, values) {
  let value = values[name];
  if (Number.isFinite(value)) return Number(value);
  const selected = FEATURE_BY_NAME.get(name);
  value = Number(selected?.imputeMedian);
  return Number.isFinite(value) ? value : 0;
}

function parameterEvidence(parameterKey, values, readiness = 1) {
  const model = PARAMETER_EVIDENCE[parameterKey];
  if (!model || !Array.isArray(model.features) || !model.features.length) {
    return { probability: 0, ratio: 0, risk: 0, usedFeatures: 0, contributions: [] };
  }
  const means = Array.isArray(model.scalerMean) ? model.scalerMean : [];
  const scales = Array.isArray(model.scalerScale) ? model.scalerScale : [];
  const coefficients = Array.isArray(model.coefficients) ? model.coefficients : [];
  let logit = Number(model.intercept) || 0;
  const contributions = [];
  let used = 0;
  model.features.forEach((name, index) => {
    const value = featureValue(name, values);
    const mean = Number(means[index]) || 0;
    const scaleRaw = Number(scales[index]);
    const scale = Number.isFinite(scaleRaw) && Math.abs(scaleRaw) > 1e-12 ? scaleRaw : 1;
    const coefficient = Number(coefficients[index]) || 0;
    const z = (value - mean) / scale;
    const contribution = coefficient * z;
    logit += contribution;
    used += 1;
    contributions.push({ name, value, z, contribution });
  });
  const probability = sigmoid(logit);
  const reference = Math.max(Number(model.normalReferenceProbability) || 1e-6, 1e-6);
  const ratio = probability / reference;
  const rawRisk = sigmoid(RISK_SLOPE * (ratio - RATIO_THRESHOLD));
  // Before the full 64-point context exists, report low confidence and suppress
  // aggressive state changes.  Normal evidence from imputed long-window features
  // must not create startup alarms.
  const risk = Math.min(MODEL_RISK_CAP, clamp(rawRisk * clamp(readiness, 0, 1)));
  return {
    probability,
    ratio,
    risk,
    usedFeatures: used,
    contributions: contributions.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)),
  };
}

function normalPdf(value, center, sigma) {
  return Math.exp(-0.5 * ((value - center) / sigma) ** 2) + 1e-8;
}

function updatePosterior(previous, evidence) {
  const predicted = [0, 0, 0];
  for (let from = 0; from < 3; from += 1) {
    for (let to = 0; to < 3; to += 1) predicted[to] += previous[from] * TRANSITION[from][to];
  }
  const likelihood = [
    normalPdf(evidence, 0.05, 0.18),
    normalPdf(evidence, 0.50, 0.23),
    normalPdf(evidence, 0.90, 0.18),
  ];
  const unnormalized = predicted.map((value, index) => value * likelihood[index]);
  const total = unnormalized.reduce((sum, value) => sum + value, 0) || 1;
  return unnormalized.map((value) => value / total);
}

function stateKey(satelliteId, subsystemId) {
  return `${satelliteId || "default"}::${subsystemId}`;
}

function updateState(satelliteId, subsystemId, evidence) {
  const key = stateKey(satelliteId, subsystemId);
  const previous = stateStore.get(key) || INITIAL_POSTERIOR;
  const posterior = updatePosterior(previous, evidence);
  stateStore.set(key, posterior);
  return posterior;
}

function scoreFromPosterior(posterior) {
  return Math.round(clamp(100 * (posterior[0] + 0.55 * posterior[1]), 0, 100));
}

function stateName(score) {
  return score >= 85 ? "健康" : score >= 60 ? "退化" : "故障";
}

function currentParameterMap(telemetry) {
  const map = new Map();
  (Array.isArray(telemetry.coreParameters) ? telemetry.coreParameters : []).forEach((parameter) => {
    if (!parameter) return;
    map.set(parameter.key, parameter);
  });
  return map;
}

function modelsForParameters(parameterKeys) {
  const set = new Set(parameterKeys);
  return SELECTED.filter((feature) => set.has(feature.parameterKey));
}

function modelsForSubsystem(subsystemId) {
  return SELECTED.filter((feature) => Boolean(PARAMETER_TO_SUBSYSTEMS[feature.parameterKey]?.[subsystemId]));
}

function calculate(input) {
  if (!input || typeof input !== "object") throw new TypeError("health input must be an object");
  const telemetry = input.telemetry && typeof input.telemetry === "object" ? input.telemetry : input;
  const satelliteId = String(input.satelliteId || telemetry.satelliteId || "default");
  const vector = extractFeatureVector(telemetry, { selectedFeatureNames: SELECTED.map((feature) => feature.name), scales: MODEL.featureSpace?.scales || [8, 24, 64] });
  const values = vector.features;
  const parameterMap = currentParameterMap(telemetry);

  const parameterAssessments = {};
  Object.keys(PARAMETER_TO_SUBSYSTEMS).forEach((parameterKey) => {
    const evidenceResult = parameterEvidence(parameterKey, values, vector.readiness);
    const parameter = parameterMap.get(parameterKey);
    const threshold = telemetry.thresholds?.[parameterKey] || {};
    const currentValue = Number(parameter?.value ?? parameter?.currentValue);
    const min = Number(threshold.min ?? threshold.lower);
    const max = Number(threshold.max ?? threshold.upper);
    const hardLimitViolation = Number.isFinite(currentValue) && (
      (Number.isFinite(min) && currentValue < min) ||
      (Number.isFinite(max) && currentValue > max)
    );
    const hardLimitRisk = hardLimitViolation ? 1 : 0;
    parameterAssessments[parameterKey] = {
      risk: Math.max(evidenceResult.risk, hardLimitRisk),
      modelRisk: evidenceResult.risk,
      hardLimitRisk,
      hardLimitViolation,
      probability: evidenceResult.probability,
      ratio: evidenceResult.ratio,
      usedFeatures: evidenceResult.usedFeatures,
      contributions: evidenceResult.contributions.slice(0, 5),
    };
  });

  const parameterRisks = Object.values(parameterAssessments).map((item) => item.risk).sort((a, b) => b - a);
  const maxParameterRisk = parameterRisks[0] || 0;
  const secondParameterRisk = parameterRisks[1] || 0;
  const w1 = Number.isFinite(TOP_WEIGHTS[0]) ? TOP_WEIGHTS[0] : 0.65;
  const w2 = Number.isFinite(TOP_WEIGHTS[1]) ? TOP_WEIGHTS[1] : 0.35;
  const globalEvidence = clamp(w1 * maxParameterRisk + w2 * Math.sqrt(maxParameterRisk * secondParameterRisk));
  const globalPosterior = updateState(satelliteId, "__global__", globalEvidence);
  const globalHealth = scoreFromPosterior(globalPosterior);

  const unitAssessments = UNIT_DEFINITIONS.map((unit) => {
    const members = unit.parameters.map((key) => ({ key, assessment: parameterAssessments[key] })).filter((item) => item.assessment);
    const risks = members.map((item) => item.assessment.risk).sort((a, b) => b - a);
    const risk = risks.length ? clamp(0.72 * risks[0] + 0.28 * (risks.length > 1 ? Math.sqrt(risks[0] * risks[1]) : risks[0])) : 0;
    const usedFeatures = members.reduce((sum, item) => sum + item.assessment.usedFeatures, 0);
    const expectedFeatures = modelsForParameters(unit.parameters).length;
    const health = Math.round(100 * (1 - risk));
    const contributions = members.flatMap((item) => item.assessment.contributions.map((feature) => ({ ...feature, parameterKey: item.key })))
      .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)).slice(0, 5);
    return {
      id: unit.id,
      key: unit.key,
      name: unit.name,
      health,
      percentage: health,
      state: stateName(health),
      risk: Number(risk.toFixed(4)),
      confidence: Number((vector.readiness * Math.min(1, usedFeatures / Math.max(1, expectedFeatures))).toFixed(4)),
      subsystems: unit.subsystems,
      critical: unit.critical,
      contributors: contributions.map((item) => {
        const model = FEATURE_BY_NAME.get(item.name);
        return {
          feature: item.name,
          parameterKey: item.parameterKey || model?.parameterKey,
          descriptor: model?.descriptor,
          scale: model?.scale,
          evidence: Number(sigmoid(item.contribution).toFixed(4)),
          contribution: Number(item.contribution.toFixed(4)),
        };
      }),
    };
  });

  const subsystemAssessments = SUBSYSTEMS.map((subsystem) => {
    const relevantUnits = unitAssessments.filter((unit) => unit.subsystems[subsystem.id]);
    const weightedTotal = relevantUnits.reduce((sum, unit) => sum + unit.subsystems[subsystem.id], 0);
    const weightedRisk = weightedTotal
      ? relevantUnits.reduce((sum, unit) => sum + unit.risk * unit.subsystems[subsystem.id], 0) / weightedTotal
      : 0;
    const criticalRisk = relevantUnits.filter((unit) => unit.critical).reduce((maximum, unit) => Math.max(maximum, unit.risk), 0);
    const sortedRisks = relevantUnits.map((unit) => unit.risk).sort((a, b) => b - a);
    const interactionRisk = sortedRisks.length > 1 ? Math.sqrt(sortedRisks[0] * sortedRisks[1]) : 0;
    const evidence = clamp(0.60 * weightedRisk + 0.28 * criticalRisk + 0.12 * interactionRisk);
    const posterior = updateState(satelliteId, subsystem.id, evidence);
    const score = scoreFromPosterior(posterior);
    const models = modelsForSubsystem(subsystem.id);
    const usedFeatures = models.filter((feature) => Number.isFinite(values[feature.name])).length;
    return {
      id: subsystem.id,
      name: subsystem.name,
      score,
      confidence: Number((vector.readiness * Math.min(1, usedFeatures / Math.max(1, models.length))).toFixed(4)),
      state: stateName(score),
      probabilities: {
        healthy: Number(posterior[0].toFixed(6)),
        degraded: Number(posterior[1].toFixed(6)),
        fault: Number(posterior[2].toFixed(6)),
      },
      evidence: Number(evidence.toFixed(4)),
      selectedFeatureCount: models.length,
      contributors: relevantUnits
        .map((unit) => ({ key: unit.key, name: unit.name, risk: unit.risk, health: unit.health, weight: unit.subsystems[subsystem.id] }))
        .sort((a, b) => b.risk * b.weight - a.risk * a.weight),
    };
  });

  let missionSuccess = 1;
  SUBSYSTEMS.forEach((subsystem) => {
    const assessment = subsystemAssessments.find((item) => item.id === subsystem.id);
    const posteriorRisk = assessment.probabilities.fault + 0.35 * assessment.probabilities.degraded;
    missionSuccess *= 1 - subsystem.criticality * posteriorRisk;
  });
  const missionScore = 100 * clamp(missionSuccess, 0, 1);
  const overallScore = Math.round(clamp(0.75 * missionScore + 0.25 * globalHealth, 0, 100));
  const selectedAvailable = SELECTED.filter((feature) => Number.isFinite(values[feature.name])).length;
  const overallConfidence = clamp(vector.readiness * selectedAvailable / SELECTED.length);

  const subsystemHealth = subsystemAssessments.map((item) => ({
    name: item.name,
    value: item.score,
    percentage: item.score,
  }));

  const signalAssessments = (Array.isArray(telemetry.coreParameters) ? telemetry.coreParameters : []).map((parameter) => {
    const assessment = parameterAssessments[parameter.key] || { risk: 0, usedFeatures: 0, contributions: [] };
    const threshold = telemetry.thresholds?.[parameter.key] || {};
    return {
      key: parameter.key,
      name: parameter.name,
      confidence: Number((vector.readiness * (assessment.usedFeatures ? 1 : 0)).toFixed(4)),
      currentRisk: Number(assessment.risk.toFixed(4)),
      modelRisk: Number((assessment.modelRisk || 0).toFixed(4)),
      hardLimitRisk: Number((assessment.hardLimitRisk || 0).toFixed(4)),
      hardLimitViolation: Boolean(assessment.hardLimitViolation),
      evidenceProbability: Number((assessment.probability || 0).toFixed(6)),
      evidenceRatio: Number((assessment.ratio || 0).toFixed(4)),
      currentValue: Number(parameter.value ?? parameter.currentValue),
      threshold,
      selectedFeatureCount: assessment.usedFeatures,
      featureContributors: assessment.contributions.map((item) => ({
        feature: item.name,
        contribution: Number(item.contribution.toFixed(4)),
      })),
    };
  });

  return {
    algorithmVersion: VERSION,
    modelVersion: MODEL.schemaVersion,
    overallScore,
    overallConfidence: Number(overallConfidence.toFixed(4)),
    overallState: stateName(overallScore),
    globalFaultProbability: Number(globalEvidence.toFixed(6)),
    rawGlobalNaiveBayesProbability: null,
    globalEvidenceProbability: Number(globalEvidence.toFixed(6)),
    globalProbabilities: {
      healthy: Number(globalPosterior[0].toFixed(6)),
      degraded: Number(globalPosterior[1].toFixed(6)),
      fault: Number(globalPosterior[2].toFixed(6)),
    },
    subsystemHealth,
    totalHealthInfo: [
      { value: overallScore, name: "健康" },
      { value: 100 - overallScore, name: "不健康" },
    ],
    childHealthInfo: [
      subsystemHealth.map((item) => item.value),
      subsystemHealth.map((item) => 100 - item.value),
    ],
    units: unitAssessments,
    subsystems: subsystemAssessments,
    signalAssessments,
    msf: {
      candidateFeatureCount: MODEL.featureSpace?.candidateCount || 288,
      selectedFeatureCount: SELECTED.length,
      selectedFeatureRatio: Number((SELECTED.length / Math.max(1, MODEL.featureSpace?.candidateCount || 288)).toFixed(4)),
      guidedCardinalities: MODEL.method?.guidedCardinalities || [],
      currentAvailableSelectedFeatures: selectedAvailable,
      windowReadiness: Number(vector.readiness.toFixed(4)),
      evidenceModel: EVIDENCE_CONFIG.model || "parameter-local logistic",
      evidenceRatioThreshold: RATIO_THRESHOLD,
      topCurrentEvidence: Object.entries(parameterAssessments)
        .sort((a, b) => b[1].risk - a[1].risk)
        .slice(0, 8)
        .map(([parameterKey, item]) => ({
          parameterKey,
          risk: Number(item.risk.toFixed(4)),
          probability: Number((item.probability || 0).toFixed(6)),
          ratio: Number((item.ratio || 0).toFixed(4)),
        })),
    },
  };
}

function reset(satelliteId) {
  const prefix = `${satelliteId || "default"}::`;
  for (const key of stateStore.keys()) {
    if (key.startsWith(prefix)) stateStore.delete(key);
  }
}

function snapshotState(satelliteId) {
  const id = String(satelliteId || 'default');
  const prefix = `${id}::`;
  const states = {};
  for (const [key, posterior] of stateStore.entries()) {
    if (!key.startsWith(prefix)) continue;
    states[key.slice(prefix.length)] = posterior.map(Number);
  }
  return {
    schemaVersion: 'health-dbn-state-1.0',
    satelliteId: id,
    algorithmVersion: VERSION,
    modelSchemaVersion: MODEL.schemaVersion || null,
    updatedAt: new Date().toISOString(),
    states,
  };
}

function restoreState(satelliteId, snapshot) {
  const id = String(satelliteId || 'default');
  reset(id);
  if (!snapshot || typeof snapshot !== 'object' || !snapshot.states || typeof snapshot.states !== 'object') return 0;
  let restored = 0;
  Object.entries(snapshot.states).forEach(([subsystemId, raw]) => {
    if (!Array.isArray(raw) || raw.length !== 3) return;
    const values = raw.map(Number);
    if (!values.every((value) => Number.isFinite(value) && value >= 0)) return;
    const total = values.reduce((sum, value) => sum + value, 0);
    if (!(total > 0)) return;
    stateStore.set(stateKey(id, subsystemId), values.map((value) => value / total));
    restored += 1;
  });
  return restored;
}

function stateCount() {
  return stateStore.size;
}

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
  definitions: { subsystems: SUBSYSTEMS, unitDefinitions: UNIT_DEFINITIONS },
};
