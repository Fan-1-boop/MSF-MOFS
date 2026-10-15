"use strict";

/**
 * V6 coupled-fault training data generator.
 *
 * One row = one causal telemetry window.
 * Candidate features = 582 (288 univariate + 294 cross-channel relation features).
 * Labels come only from injected point/coupled fault ground truth.
 */

const fs = require("node:fs");
const path = require("node:path");
const {
  TelemetrySimulator,
  PARAMETERS,
  FAULT_MODES,
  COUPLED_FAULT_MODES,
} = require("./telemetry-simulator.js");
const {
  DEFAULT_SCALES,
  RELATION_PAIRS,
  extractFeatureVector,
  expectedFeatureNames,
} = require("../health-v2-service/core/health-features.js");

const CONTINUOUS = new Set(["drift", "step", "noise", "oscillation"]);
const TRANSIENT = new Set(["spike", "intermittent"]);

// One moderate severity per univariate mode. The V6 focus is not to inflate
// single-channel faults but to mix them with relation-only coupled faults.
const UNI_MAGNITUDE = {
  drift: 0.0035,
  spike: 0.40,
  step: 0.12,
  noise: 0.035,
  oscillation: 0.10,
  intermittent: 0.40,
};
const COUPLED_SEVERITIES = [0.65, 1.0];
const CRITICAL_COUPLED_PAIR_IDS = new Set(["p01_p02", "p03_p04", "p06_p12", "p10_p11", "p01_p08"]);
const COUPLED_SCENARIO_PAIRS = RELATION_PAIRS.filter((pair) => CRITICAL_COUPLED_PAIR_IDS.has(pair.id));

function escapeCsv(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  if (!/[",\n\r]/.test(text)) return text;
  return `"${text.replace(/"/g, '""')}"`;
}

function parseArgs() {
  const args = process.argv.slice(2);
  function value(name, fallback) {
    const index = args.indexOf(name);
    return index >= 0 && index + 1 < args.length ? args[index + 1] : fallback;
  }
  return {
    out: path.resolve(value("--out", path.join(__dirname, "..", "msf-training", "telemetry-msf-v6.csv"))),
    length: Number(value("--length", "224")),
    stride: Number(value("--stride", "8")),
    normalPerSplit: Number(value("--normal-per-split", "16")),
  };
}

function scenarioSeed(splitIndex, block, item, variant = 0) {
  return 2600000 + splitIndex * 300000 + block * 7000 + item * 73 + variant * 997;
}

function classifyWindow(truthWindow) {
  const recent8 = truthWindow.slice(-8);
  const transientHit = recent8.some((truth) =>
    (truth.faults || []).some((fault) => TRANSIENT.has(fault.mode)),
  );
  if (transientHit) return { label: 1, state: "abnormal-transient" };

  const flags = truthWindow.map((truth) => {
    const uni = (truth.faults || []).some((fault) => CONTINUOUS.has(fault.mode));
    const coupled = (truth.coupledFaults || []).length > 0;
    return uni || coupled ? 1 : 0;
  });
  const ratio = flags.reduce((sum, value) => sum + value, 0) / Math.max(1, flags.length);
  if (ratio >= 0.5) return { label: 1, state: "abnormal" };
  if (ratio > 0) return { label: -1, state: "transition" };
  return { label: 0, state: "normal" };
}

function buildScenario({ split, id, seed, fault = null, coupledFault = null, length, stride }) {
  const noiseScale = 0.82 + ((seed % 11) / 28);      // ~0.82 .. 1.18
  const amplitudeScale = 0.84 + ((seed % 9) / 25);  // ~0.84 .. 1.16
  const simulator = new TelemetrySimulator({
    seed,
    faults: fault ? [fault] : [],
    coupledFaults: coupledFault ? [coupledFault] : [],
    scenarioId: id,
    windowSize: Math.max(...DEFAULT_SCALES),
    stepMs: 1000,
    noiseScale,
    amplitudeScale,
  });
  simulator.satelliteId = `SIM-${split.toUpperCase()}`;
  simulator.satelliteName = `MSF-V6-${split}`;

  const truthHistory = [];
  const rows = [];
  const baseTime = Date.UTC(2026, 0, 1) + seed * 1000;
  for (let tick = 0; tick < length; tick += 1) {
    const packet = simulator.nextPoint(baseTime);
    truthHistory.push(packet.groundTruth);
    if (tick < Math.max(...DEFAULT_SCALES) - 1 || tick % stride !== 0) continue;

    const vector = extractFeatureVector(packet.telemetry);
    const truthWindow = truthHistory.slice(-Math.max(...DEFAULT_SCALES));
    const cls = classifyWindow(truthWindow);
    rows.push({
      scenarioId: id,
      split,
      tick,
      state: cls.state,
      label: cls.label,
      faultScope: coupledFault ? "coupled" : (fault ? "univariate" : "normal"),
      faultKey: fault?.key || "",
      faultPair: coupledFault ? `${coupledFault.a}|${coupledFault.b}` : "",
      faultMode: coupledFault?.mode || fault?.mode || "normal",
      magnitude: coupledFault?.severity ?? fault?.magnitude ?? 0,
      startTick: coupledFault?.startTick ?? fault?.startTick ?? "",
      readiness: vector.readiness,
      features: vector.features,
    });
  }
  return rows;
}

function main() {
  const cfg = parseArgs();
  fs.mkdirSync(path.dirname(cfg.out), { recursive: true });
  const parameterKeys = PARAMETERS.map((parameter) => parameter.key);
  const featureNames = expectedFeatureNames(parameterKeys, DEFAULT_SCALES, { includeRelations: true });
  const metaColumns = [
    "scenarioId", "split", "tick", "state", "label", "faultScope", "faultKey", "faultPair", "faultMode",
    "magnitude", "startTick", "readiness",
  ];
  const stream = fs.createWriteStream(cfg.out, { encoding: "utf8" });
  stream.write([...metaColumns, ...featureNames].join(",") + "\n");

  const splits = ["train", "val", "test"];
  let scenarioCount = 0;
  let rowCount = 0;
  const labelCounts = { "-1": 0, "0": 0, "1": 0 };
  const scopeCounts = { normal: 0, univariate: 0, coupled: 0 };

  function emit(rows) {
    rows.forEach((row) => {
      const meta = metaColumns.map((column) => row[column]);
      const featureValues = featureNames.map((name) => {
        const value = row.features[name];
        return Number.isFinite(value) ? Number(value.toPrecision(10)) : "";
      });
      stream.write([...meta, ...featureValues].map(escapeCsv).join(",") + "\n");
      rowCount += 1;
      labelCounts[String(row.label)] += 1;
    });
  }

  splits.forEach((split, splitIndex) => {
    for (let i = 0; i < cfg.normalPerSplit; i += 1) {
      const seed = scenarioSeed(splitIndex, 1, i + 1);
      const id = `${split}-normal-${String(i + 1).padStart(2, "0")}`;
      emit(buildScenario({ split, id, seed, length: cfg.length, stride: cfg.stride }));
      scenarioCount += 1; scopeCounts.normal += 1;
    }

    PARAMETERS.forEach((parameter, parameterIndex) => {
      FAULT_MODES.forEach((mode, modeIndex) => {
        const seed = scenarioSeed(splitIndex, 10 + parameterIndex, modeIndex + 1);
        const startTick = 72 + (seed % 17);
        const fault = {
          key: parameter.key,
          mode,
          magnitude: UNI_MAGNITUDE[mode],
          startTick,
          endTick: cfg.length + 1,
        };
        const id = `${split}-uni-${parameter.key}-${mode}`;
        emit(buildScenario({ split, id, seed, fault, length: cfg.length, stride: cfg.stride }));
        scenarioCount += 1; scopeCounts.univariate += 1;
      });
    });

    COUPLED_SCENARIO_PAIRS.forEach((pair, pairIndex) => {
      COUPLED_FAULT_MODES.forEach((mode, modeIndex) => {
        COUPLED_SEVERITIES.forEach((severity, severityIndex) => {
          const seed = scenarioSeed(splitIndex, 100 + pairIndex, modeIndex + 1, severityIndex + 1);
          const startTick = 72 + (seed % 17);
          const coupledFault = {
            a: pair.a,
            b: pair.b,
            target: pair.b,
            group: pair.group,
            mode,
            severity,
            lag: 8 + 4 * severityIndex,
            startTick,
            endTick: cfg.length + 1,
          };
          const id = `${split}-coupled-${pair.id}-${mode}-s${severityIndex + 1}`;
          emit(buildScenario({ split, id, seed, coupledFault, length: cfg.length, stride: cfg.stride }));
          scenarioCount += 1; scopeCounts.coupled += 1;
        });
      });
    });
  });

  stream.end(() => {
    const manifest = {
      schemaVersion: "msf-telemetry-dataset-v6-2.0",
      generatedAt: new Date().toISOString(),
      file: path.basename(cfg.out),
      scenarios: scenarioCount,
      rows: rowCount,
      featureCount: featureNames.length,
      featureBreakdown: { univariate: 288, crossChannel: featureNames.length - 288 },
      relationPairs: RELATION_PAIRS.length,
      coupledScenarioPairs: COUPLED_SCENARIO_PAIRS.map((pair) => pair.id),
      coupledFaultModes: COUPLED_FAULT_MODES,
      scales: DEFAULT_SCALES,
      labelCounts,
      scenarioScopeCounts: scopeCounts,
      labelDefinition: {
        normal: "0 = no continuous/coupled fault in 64-point window and no transient univariate event in last 8 points",
        abnormal: "1 = continuous/coupled fault active in >=50% of the 64-point window OR transient event in last 8 points",
        transition: "-1 = continuous/coupled fault active in >0 and <50% of the 64-point window; excluded from MSF training",
      },
      splitRule: "scenario-level split; no window from one scenario crosses train/val/test",
      benchmarkBoundary: "synthetic coupling benchmark; relation groups are controlled engineering-inspired abstractions, not flight-validated equations",
    };
    fs.writeFileSync(`${cfg.out}.manifest.json`, JSON.stringify(manifest, null, 2) + "\n", "utf8");
    console.log(JSON.stringify(manifest, null, 2));
  });
}

if (require.main === module) main();
module.exports = { buildScenario, classifyWindow, UNI_MAGNITUDE, COUPLED_SEVERITIES };
