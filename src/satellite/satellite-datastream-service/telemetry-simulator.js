"use strict";

/**
 * V6 satellite telemetry simulator.
 *
 * Adds two capabilities to the V5 point-fault simulator:
 * 1) normal shared latent factors that create stable subsystem relationships;
 * 2) coupled faults that alter a relationship while largely preserving the
 *    marginal waveform of the affected channel.
 *
 * This is a controlled synthetic benchmark, not a flight-physics simulator.
 */

const PARAMETERS = [
  { key: "param_1",  name: "母线电压",       unit: "V",     base: 28.25, min: 27.8, max: 28.7, amplitude: 0.006, step: 0.0004 },
  { key: "param_2",  name: "蓄电池电流",     unit: "A",     base: 4.2,   min: 0,    max: 8.5,  amplitude: 0.06,  step: 0.004 },
  { key: "param_3",  name: "蓄电池温度",     unit: "℃",    base: 20.0,  min: -5,   max: 36,   amplitude: 0.28,  step: 0.018 },
  { key: "param_4",  name: "载荷温度",       unit: "℃",    base: 28.1,  min: -10,  max: 45,   amplitude: 0.30,  step: 0.02 },
  { key: "param_5",  name: "推力器压力",     unit: "MPa",   base: 1.80,  min: 1.55, max: 2.05, amplitude: 0.005, step: 0.0003 },
  { key: "param_6",  name: "动量轮转速",     unit: "rpm",   base: 4000,  min: 0,    max: 5200, amplitude: 32,    step: 2.2 },
  { key: "param_7",  name: "陀螺漂移",       unit: "°/h",   base: 0.02,  min: 0,    max: 0.08, amplitude: 0.0005,step: 0.00004 },
  { key: "param_8",  name: "SADA角度",       unit: "°",     base: 45,    min: 0,    max: 88,   amplitude: 0.4,   step: 0.03 },
  { key: "param_9",  name: "电源控制器温度", unit: "℃",    base: 31,    min: -10,  max: 50,   amplitude: 0.34,  step: 0.022 },
  { key: "param_10", name: "星载计算机负载", unit: "%",     base: 52,    min: 0,    max: 85,   amplitude: 0.7,   step: 0.05 },
  { key: "param_11", name: "下行速率",       unit: "Mbps",  base: 132,   min: 30,   max: 190,  amplitude: 0.9,   step: 0.06 },
  { key: "param_12", name: "姿态误差",       unit: "°",     base: 0.10,  min: 0,    max: 0.5,  amplitude: 0.002, step: 0.00016 },
];

const TWO_PI = Math.PI * 2;
const FAULT_MODES = ["drift", "spike", "step", "noise", "oscillation", "intermittent"];
const TRANSIENT_MODES = new Set(["spike", "intermittent"]);
const COUPLED_FAULT_MODES = ["invert", "lag", "decouple"];

const SHARED_GROUPS = {
  power: { keys: ["param_1", "param_2", "param_9", "param_10"], period: 37, gain: 2.10 },
  thermal: { keys: ["param_3", "param_4", "param_9"], period: 53, gain: 2.00 },
  attitude: { keys: ["param_6", "param_7", "param_12"], period: 29, gain: 2.20 },
  communication: { keys: ["param_10", "param_11"], period: 43, gain: 2.05 },
  solar: { keys: ["param_1", "param_8"], period: 61, gain: 1.90 },
};

function mulberry32(seed) {
  let state = seed >>> 0;
  return function next() {
    state |= 0;
    state = (state + 0x6D2B79F5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function normalizeMode(mode) {
  return FAULT_MODES.includes(mode) ? mode : "drift";
}

function normalizeCoupledMode(mode) {
  return COUPLED_FAULT_MODES.includes(mode) ? mode : "invert";
}

function finiteOr(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function parseFault(raw) {
  if (!raw || raw === "off" || raw === "none") return null;
  if (typeof raw === "object") {
    const startTick = Math.max(0, Math.round(finiteOr(raw.startTick, 0)));
    const endCandidate = finiteOr(raw.endTick, Infinity);
    const magnitude = finiteOr(raw.magnitude, 0);
    const magnitudeStart = Number.isFinite(Number(raw.magnitudeStart)) ? Number(raw.magnitudeStart) : magnitude;
    const magnitudeEnd = Number.isFinite(Number(raw.magnitudeEnd)) ? Number(raw.magnitudeEnd) : magnitude;
    return {
      key: String(raw.key || "param_4"), mode: normalizeMode(raw.mode), magnitude, magnitudeStart, magnitudeEnd, startTick,
      endTick: Number.isFinite(endCandidate) ? Math.max(startTick + 1, Math.round(endCandidate)) : Infinity,
    };
  }
  const [keyPart, mode = "drift", magnitudePart, startPart, endPart] = String(raw).split(":");
  const startTick = Math.max(0, Math.round(finiteOr(startPart, 0)));
  const endCandidate = finiteOr(endPart, Infinity);
  const magnitude = finiteOr(magnitudePart, 0);
  return {
    key: keyPart || "param_4", mode: normalizeMode(mode), magnitude, magnitudeStart: magnitude, magnitudeEnd: magnitude, startTick,
    endTick: Number.isFinite(endCandidate) ? Math.max(startTick + 1, Math.round(endCandidate)) : Infinity,
  };
}

function parseFaults(raw) {
  if (raw === undefined || raw === null || raw === "off" || raw === "none") return [];
  if (Array.isArray(raw)) return raw.map(parseFault).filter(Boolean);
  const single = parseFault(raw);
  return single ? [single] : [];
}

function commonGroup(a, b) {
  return Object.entries(SHARED_GROUPS).find(([, cfg]) => cfg.keys.includes(a) && cfg.keys.includes(b))?.[0] || null;
}

function parseCoupledFault(raw) {
  if (!raw || raw === "off" || raw === "none") return null;
  const source = typeof raw === "object" ? raw : {};
  const pair = Array.isArray(source.pair) ? source.pair : [source.a, source.b];
  const a = String(pair[0] || "param_1");
  const b = String(pair[1] || "param_2");
  const startTick = Math.max(0, Math.round(finiteOr(source.startTick, 0)));
  const endCandidate = finiteOr(source.endTick, Infinity);
  return {
    a,
    b,
    target: String(source.target || b),
    group: String(source.group || commonGroup(a, b) || "power"),
    mode: normalizeCoupledMode(source.mode),
    severity: Math.max(0, Math.min(1.5, finiteOr(source.severity, 1))),
    lag: Math.max(1, Math.round(finiteOr(source.lag, 8))),
    startTick,
    endTick: Number.isFinite(endCandidate) ? Math.max(startTick + 1, Math.round(endCandidate)) : Infinity,
  };
}

function parseCoupledFaults(raw) {
  if (raw === undefined || raw === null || raw === "off" || raw === "none") return [];
  if (Array.isArray(raw)) return raw.map(parseCoupledFault).filter(Boolean);
  const single = parseCoupledFault(raw);
  return single ? [single] : [];
}

class TelemetrySimulator {
  constructor(options = {}) {
    this.seed = Number(options.seed) || 20260814;
    this.faults = parseFaults(options.faults !== undefined ? options.faults : options.fault);
    this.coupledFaults = parseCoupledFaults(options.coupledFaults !== undefined ? options.coupledFaults : options.coupledFault);
    this.amplitudeScale = Number(options.amplitudeScale) || 1;
    this.noiseScale = Number(options.noiseScale) || 1;
    this.windowSize = Number(options.windowSize) || 64;
    this.stepMs = Number(options.stepMs) || 1000;
    this.scenarioId = String(options.scenarioId || "");
    this.params = this._buildParams();
    this.random = mulberry32(this.seed);
    this.phases = this.params.map(() => this.random() * TWO_PI);
    this.groupPhases = Object.fromEntries(Object.keys(SHARED_GROUPS).map((name) => [name, this.random() * TWO_PI]));
    this.driftAccumulators = {};
    this.epochMs = null;
    this.clockMs = null;
    this.tick = 0;
    this.windowBuffer = this.params.map(() => ({ axisDataX: [], seriesData: [] }));
  }

  _buildParams() {
    return PARAMETERS.map((p) => ({ ...p, amplitude: p.amplitude * this.amplitudeScale, step: p.step * this.noiseScale }));
  }

  reconfigure(patch = {}) {
    if (patch.seed !== undefined) this.seed = Number(patch.seed) || 20260814;
    if (patch.amplitudeScale !== undefined) this.amplitudeScale = Number(patch.amplitudeScale) || 1;
    if (patch.noiseScale !== undefined) this.noiseScale = Number(patch.noiseScale) || 1;
    if (patch.faults !== undefined) this.faults = parseFaults(patch.faults);
    else if (patch.fault !== undefined) this.faults = parseFaults(patch.fault);
    if (patch.coupledFaults !== undefined) this.coupledFaults = parseCoupledFaults(patch.coupledFaults);
    else if (patch.coupledFault !== undefined) this.coupledFaults = parseCoupledFaults(patch.coupledFault);
    if (patch.windowSize !== undefined) this.windowSize = Number(patch.windowSize) || 64;
    if (patch.stepMs !== undefined) this.stepMs = Number(patch.stepMs) || 1000;
    if (patch.scenarioId !== undefined) this.scenarioId = String(patch.scenarioId || "");
    this.params = this._buildParams();
    this.random = mulberry32(this.seed);
    this.phases = this.params.map(() => this.random() * TWO_PI);
    this.groupPhases = Object.fromEntries(Object.keys(SHARED_GROUPS).map((name) => [name, this.random() * TWO_PI]));
    this.driftAccumulators = {};
    this.epochMs = null;
    this.clockMs = null;
    this.tick = 0;
    this.windowBuffer = this.params.map(() => ({ axisDataX: [], seriesData: [] }));
  }

  _faultScheduled(fault, tick) { return tick >= fault.startTick && tick < fault.endTick; }

  _faultMagnitude(fault, tick) {
    const start = Number.isFinite(Number(fault.magnitudeStart)) ? Number(fault.magnitudeStart) : Number(fault.magnitude) || 0;
    const end = Number.isFinite(Number(fault.magnitudeEnd)) ? Number(fault.magnitudeEnd) : Number(fault.magnitude) || 0;
    if (!Number.isFinite(fault.endTick) || fault.endTick <= fault.startTick + 1 || start === end) return start;
    const alpha = Math.min(1, Math.max(0, (tick - fault.startTick) / (fault.endTick - fault.startTick - 1)));
    return start + alpha * (end - start);
  }

  _groupFactor(group, tick, phaseOffset = 0) {
    const cfg = SHARED_GROUPS[group];
    if (!cfg) return 0;
    const phase = (this.groupPhases[group] || 0) + phaseOffset;
    return Math.sin(TWO_PI * (tick / cfg.period) + phase) + 0.28 * Math.sin(TWO_PI * (tick / (cfg.period * 0.43)) + phase * 1.71);
  }

  _groupContribution(p, group, tick, phaseOffset = 0) {
    const cfg = SHARED_GROUPS[group];
    if (!cfg || !cfg.keys.includes(p.key)) return 0;
    const memberships = Object.values(SHARED_GROUPS).filter((item) => item.keys.includes(p.key)).length;
    return this._groupFactor(group, tick, phaseOffset) * p.amplitude * cfg.gain / Math.sqrt(Math.max(1, memberships));
  }

  nextPoint(baseTimeMs) {
    if (!this.epochMs) this.epochMs = baseTimeMs;
    if (!this.clockMs) this.clockMs = baseTimeMs;
    const t = this.tick++;
    const timestamp = new Date(this.clockMs).toISOString().slice(0, 19).replace("T", " ");

    const coreParameters = [];
    const trendCharts = [];
    const thresholds = {};
    const pointFaults = [];
    const pointCoupledFaults = [];

    this.params.forEach((p, index) => {
      const phase = this.phases[index];
      const slow = Math.sin(TWO_PI * (t / 60) + phase) * p.amplitude;
      const mid = Math.sin(TWO_PI * (t / 17) + phase * 1.7) * p.amplitude * 0.45;
      const noise = (this.random() - 0.5) * 2 * p.step * 1.6;
      let value = p.base + slow + mid + noise;
      Object.keys(SHARED_GROUPS).forEach((group) => { value += this._groupContribution(p, group, t); });

      const span = Math.max(1e-9, p.max - p.min);
      const direction = p.key === "param_11" ? -1 : 1;

      // Coupled faults only alter the target channel's shared latent component.
      // Full inversion keeps the latent component's marginal amplitude but reverses its relation.
      this.coupledFaults.forEach((fault) => {
        if (fault.target !== p.key || !this._faultScheduled(fault, t)) return;
        const current = this._groupContribution(p, fault.group, t);
        let replacement = current;
        if (fault.mode === "invert") {
          replacement = -current;
        } else if (fault.mode === "lag") {
          replacement = this._groupContribution(p, fault.group, t - fault.lag);
        } else if (fault.mode === "decouple") {
          replacement = this._groupContribution(p, fault.group, t, 1.937);
        }
        const effect = fault.severity * (replacement - current);
        value += effect;
        pointCoupledFaults.push({
          pair: [fault.a, fault.b],
          group: fault.group,
          target: fault.target,
          mode: fault.mode,
          severity: fault.severity,
          lag: fault.lag,
          startTick: fault.startTick,
          endTick: Number.isFinite(fault.endTick) ? fault.endTick : null,
          age: t - fault.startTick,
          injectedEffect: effect,
          normalizedEffect: Math.abs(effect) / span,
        });
      });

      this.faults.forEach((fault) => {
        if (fault.key !== p.key || !this._faultScheduled(fault, t)) return;
        const currentMagnitude = this._faultMagnitude(fault, t);
        let effect = 0;
        let triggered = !TRANSIENT_MODES.has(fault.mode);
        switch (fault.mode) {
          case "drift": {
            const perTick = (currentMagnitude || 0.002) * span;
            this.driftAccumulators[p.key] = (this.driftAccumulators[p.key] || 0) + perTick;
            effect = this.driftAccumulators[p.key]; value += effect; break;
          }
          case "spike": {
            if ((t - fault.startTick) % 30 === 0) { triggered = true; effect = direction * (currentMagnitude || 0.5) * span; value += effect; }
            break;
          }
          case "step": { effect = direction * (currentMagnitude || 0.2) * span; value += effect; break; }
          case "noise": { effect = (this.random() - 0.5) * 2 * (currentMagnitude || 0.03) * span; value += effect; break; }
          case "oscillation": { effect = Math.sin(TWO_PI * ((t - fault.startTick) / 6)) * (currentMagnitude || 0.15) * span; value += effect; break; }
          case "intermittent": {
            if (this.random() < 0.2) { triggered = true; effect = direction * (currentMagnitude || 0.5) * span; value += effect; }
            break;
          }
          default: break;
        }
        if (triggered) {
          pointFaults.push({
            key: p.key, name: p.name, mode: fault.mode, magnitude: currentMagnitude, configuredMagnitude: fault.magnitude,
            startTick: fault.startTick, endTick: Number.isFinite(fault.endTick) ? fault.endTick : null,
            age: t - fault.startTick, transient: TRANSIENT_MODES.has(fault.mode), injectedEffect: effect,
            normalizedEffect: Math.abs(effect) / span,
          });
        }
      });

      value = Number(value.toFixed(6));
      const buffer = this.windowBuffer[index];
      buffer.axisDataX.push(timestamp);
      buffer.seriesData.push(value);
      if (buffer.axisDataX.length > this.windowSize) { buffer.axisDataX.shift(); buffer.seriesData.shift(); }
      coreParameters.push({ key: p.key, name: p.name, value, unit: p.unit });
      trendCharts.push({ key: p.key, name: p.name, unit: p.unit, axisDataX: buffer.axisDataX.slice(), seriesData: buffer.seriesData.slice() });
      thresholds[p.key] = { name: p.name, unit: p.unit, min: p.min, max: p.max };
    });

    this.clockMs += this.stepMs;
    const active = pointFaults.length > 0 || pointCoupledFaults.length > 0;
    return {
      satelliteId: this.satelliteId,
      satelliteName: this.satelliteName,
      scenarioId: this.scenarioId || undefined,
      telemetry: { coreParameters, trendCharts, thresholds },
      groundTruth: {
        scenarioId: this.scenarioId || null,
        tick: t,
        timestamp,
        label: active ? 1 : 0,
        faultActive: active,
        faults: pointFaults,
        coupledFaults: pointCoupledFaults,
        configuredFaults: this.faults.map((fault) => ({ ...fault, endTick: Number.isFinite(fault.endTick) ? fault.endTick : null })),
        configuredCoupledFaults: this.coupledFaults.map((fault) => ({ ...fault, endTick: Number.isFinite(fault.endTick) ? fault.endTick : null })),
      },
    };
  }

  describe() {
    const uni = this.faults.length ? `${this.faults.length} univariate faults` : "no univariate fault";
    const coupled = this.coupledFaults.length ? `${this.coupledFaults.length} coupled faults` : "no coupled fault";
    return `V6 simulator seed=${this.seed}, 12 channels, window=${this.windowSize}, ${uni}, ${coupled}`;
  }
}

module.exports = {
  TelemetrySimulator,
  PARAMETERS,
  mulberry32,
  parseFaults,
  parseCoupledFaults,
  FAULT_MODES,
  TRANSIENT_MODES,
  COUPLED_FAULT_MODES,
  SHARED_GROUPS,
  commonGroup,
};
