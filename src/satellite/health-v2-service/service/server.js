"use strict";

const crypto = require("node:crypto");
const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");
const { URL } = require("node:url");
const algorithm = require("../core/health-algorithm.js");

const DEFAULT_HOST = process.env.HEALTH_V2_HOST || "127.0.0.1";
const DEFAULT_PORT = Number(process.env.HEALTH_V2_PORT || 18098);
const DEFAULT_STORAGE = path.resolve(
  process.env.HEALTH_V2_STORAGE || path.join(__dirname, "..", "storage"),
);
const MAX_JSON_BYTES = Number(process.env.HEALTH_V2_MAX_BODY || 16 * 1024 * 1024);
const MAX_IMAGE_BYTES = 12 * 1024 * 1024;
const ALGORITHM_WINDOW_SIZE = Math.max(8, Number(process.env.HEALTH_WINDOW_SIZE || 64));
const MODE_RAW = String(process.env.HEALTH_RUNTIME_MODE || "online").trim().toLowerCase();
const RUNTIME_MODE = ["online", "training", "hybrid"].includes(MODE_RAW) ? MODE_RAW : "online";

class HttpError extends Error {
  constructor(status, message, code = "REQUEST_ERROR") {
    super(message);
    this.status = status;
    this.code = code;
  }
}

function ensureDirectories(storageRoot) {
  ["health", "history", "state", "training", "layouts", "front-views"].forEach((name) => {
    fs.mkdirSync(path.join(storageRoot, name), { recursive: true });
  });
}

function satelliteKey(satelliteId) {
  const id = String(satelliteId || "").trim();
  if (!id) throw new HttpError(400, "缺少卫星ID", "SATELLITE_ID_REQUIRED");
  if (id.length > 200) throw new HttpError(400, "卫星ID过长", "SATELLITE_ID_INVALID");
  return crypto.createHash("sha256").update(id, "utf8").digest("hex").slice(0, 40);
}

function cloneJson(value) {
  return JSON.parse(JSON.stringify(value));
}

function atomicWriteJson(filePath, value) {
  const temporary = `${filePath}.${process.pid}.${Date.now()}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  fs.renameSync(temporary, filePath);
}

function readJsonFile(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch (error) {
    if (error && error.code === "ENOENT") return null;
    throw error;
  }
}

function appendJsonLine(filePath, value) {
  fs.appendFileSync(filePath, `${JSON.stringify(value)}\n`, "utf8");
}

function removeIfExists(filePath) {
  try { fs.unlinkSync(filePath); } catch (error) { if (!error || error.code !== "ENOENT") throw error; }
}

function latestHistoryPointCount(history) {
  return (Array.isArray(history?.trendCharts) ? history.trendCharts : []).reduce((max, chart) => (
    Math.max(max, Array.isArray(chart?.seriesData) ? chart.seriesData.length : 0)
  ), 0);
}

function normalizeHistoryTimestamp(value) {
  const source = String(value ?? "").trim();
  if (!source) return null;
  const match = source.match(
    /^(\d{4})[-/](\d{2})[-/](\d{2})(?:[ T/])(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?$/,
  );
  if (match) {
    const [, year, month, day, hour, minute, second] = match;
    const timestamp = Date.UTC(
      Number(year),
      Number(month) - 1,
      Number(day),
      Number(hour),
      Number(minute),
      Number(second),
    );
    const date = new Date(timestamp);
    const valid = date.getUTCFullYear() === Number(year)
      && date.getUTCMonth() + 1 === Number(month)
      && date.getUTCDate() === Number(day)
      && date.getUTCHours() === Number(hour)
      && date.getUTCMinutes() === Number(minute)
      && date.getUTCSeconds() === Number(second);
    return valid ? `${year}-${month}-${day} ${hour}:${minute}:${second}` : null;
  }
  const parsed = Date.parse(source);
  return Number.isFinite(parsed)
    ? new Date(parsed).toISOString().slice(0, 19).replace("T", " ")
    : null;
}

function historyTimestampValue(value) {
  const normalized = normalizeHistoryTimestamp(value);
  return normalized ? Date.parse(`${normalized.replace(" ", "T")}Z`) : NaN;
}

function finiteNumber(value) {
  return value !== null && value !== "" && Number.isFinite(Number(value));
}

function resolveHistoryThreshold(telemetry, chart, index) {
  const parameters = Array.isArray(telemetry.coreParameters) ? telemetry.coreParameters : [];
  const parameter = parameters.find((item) => item && (
    item.key === chart.key || item.name === chart.name
  )) || parameters[index] || {};
  const thresholds = telemetry.thresholds && typeof telemetry.thresholds === "object"
    ? telemetry.thresholds
    : {};
  const raw = thresholds[chart.key] || thresholds[chart.name] || parameter.threshold || {};
  const minimum = raw.min ?? raw.lower ?? parameter.minValue ?? parameter.min;
  const maximum = raw.max ?? raw.upper ?? parameter.maxValue ?? parameter.max;
  return {
    min: finiteNumber(minimum) ? Number(minimum) : null,
    max: finiteNumber(maximum) ? Number(maximum) : null,
    unit: String(chart.unit || parameter.unit || raw.unit || ""),
  };
}

function normalizeHistoryCharts(telemetry) {
  const incomingCharts = Array.isArray(telemetry.trendCharts) ? telemetry.trendCharts : [];
  const parameters = Array.isArray(telemetry.coreParameters) ? telemetry.coreParameters : [];
  const fallbackTimestamp = normalizeHistoryTimestamp(telemetry.timestamp || telemetry.sampleTime || telemetry.time || new Date().toISOString());
  const charts = incomingCharts.length ? incomingCharts : parameters.map((parameter, index) => ({
    key: parameter?.key || `param_${index + 1}`,
    name: parameter?.name || parameter?.key || `param_${index + 1}`,
    unit: parameter?.unit || "",
    axisDataX: [fallbackTimestamp],
    seriesData: [parameter?.value ?? parameter?.currentValue],
  }));
  return charts.map((chart, index) => {
    const source = chart && typeof chart === "object" ? chart : {};
    const key = String(source.key || `param_${index + 1}`).trim().slice(0, 200);
    const name = String(source.name || key).trim().slice(0, 200);
    const axisDataX = Array.isArray(source.axisDataX)
      ? source.axisDataX
      : (Array.isArray(source.xData) ? source.xData : [fallbackTimestamp]);
    const seriesData = Array.isArray(source.seriesData)
      ? source.seriesData
      : (Array.isArray(source.data) ? source.data : []);
    const pointCount = Math.min(axisDataX.length, seriesData.length);
    const points = new Map();
    for (let position = 0; position < pointCount; position += 1) {
      const timestamp = normalizeHistoryTimestamp(axisDataX[position]);
      if (!timestamp || !finiteNumber(seriesData[position])) continue;
      points.set(timestamp, Number(seriesData[position]));
    }
    const threshold = resolveHistoryThreshold(telemetry, { key, name, unit: source.unit }, index);
    return {
      key,
      name,
      unit: threshold.unit,
      threshold: { min: threshold.min, max: threshold.max },
      points,
    };
  }).filter((chart) => chart.key && chart.points.size);
}

function buildAlgorithmTelemetry(incomingTelemetry, history, windowSize = ALGORITHM_WINDOW_SIZE) {
  const source = cloneJson(incomingTelemetry);
  const historyCharts = Array.isArray(history?.trendCharts) ? history.trendCharts : [];
  const currentParameters = Array.isArray(source.coreParameters) ? source.coreParameters : [];
  const parameterByKey = new Map(currentParameters.map((parameter) => [String(parameter?.key || parameter?.name || ""), parameter]));
  const trendCharts = historyCharts.map((chart) => {
    const axis = Array.isArray(chart.axisDataX) ? chart.axisDataX.slice(-windowSize) : [];
    const series = Array.isArray(chart.seriesData) ? chart.seriesData.slice(-windowSize) : [];
    return {
      key: chart.key,
      name: chart.name,
      unit: chart.unit || "",
      axisDataX: axis,
      seriesData: series,
    };
  });
  const thresholds = {};
  Object.entries(history?.thresholds || {}).forEach(([key, threshold]) => {
    thresholds[key] = {
      name: threshold?.name || parameterByKey.get(key)?.name || key,
      unit: threshold?.unit || parameterByKey.get(key)?.unit || "",
      min: finiteNumber(threshold?.min) ? Number(threshold.min) : null,
      max: finiteNumber(threshold?.max) ? Number(threshold.max) : null,
    };
  });
  return {
    ...source,
    coreParameters: currentParameters,
    trendCharts,
    thresholds: Object.keys(thresholds).length ? thresholds : (source.thresholds || {}),
  };
}

function buildAnomalyEvents(trendCharts, thresholds) {
  const events = [];
  trendCharts.forEach((chart) => {
    const threshold = thresholds[chart.key] || {};
    const minimum = finiteNumber(threshold.min) ? Number(threshold.min) : null;
    const maximum = finiteNumber(threshold.max) ? Number(threshold.max) : null;
    let previousAbnormal = false;
    chart.seriesData.forEach((rawValue, index) => {
      const value = Number(rawValue);
      const below = minimum !== null && value < minimum;
      const above = maximum !== null && value > maximum;
      const abnormal = below || above;
      if (abnormal && !previousAbnormal) {
        events.push({
          timestamp: chart.axisDataX[index],
          parameterKey: chart.key,
          parameterName: chart.name,
          value,
          unit: chart.unit || threshold.unit || "",
          min: minimum,
          max: maximum,
          direction: below ? "below-min" : "above-max",
        });
      }
      previousAbnormal = abnormal;
    });
  });
  return events.sort((left, right) => (
    historyTimestampValue(left.timestamp) - historyTimestampValue(right.timestamp)
      || left.parameterKey.localeCompare(right.parameterKey)
  ));
}

function mergeTelemetryHistory(existing, satelliteId, telemetry) {
  const normalizedCharts = normalizeHistoryCharts(telemetry);
  const chartMap = new Map();
  const thresholds = {};

  if (existing && Array.isArray(existing.trendCharts)) {
    existing.trendCharts.forEach((chart) => {
      const points = new Map();
      const axisDataX = Array.isArray(chart.axisDataX) ? chart.axisDataX : [];
      const seriesData = Array.isArray(chart.seriesData) ? chart.seriesData : [];
      const count = Math.min(axisDataX.length, seriesData.length);
      for (let index = 0; index < count; index += 1) {
        const timestamp = normalizeHistoryTimestamp(axisDataX[index]);
        if (timestamp && finiteNumber(seriesData[index])) points.set(timestamp, Number(seriesData[index]));
      }
      chartMap.set(String(chart.key), {
        key: String(chart.key),
        name: String(chart.name || chart.key),
        unit: String(chart.unit || ""),
        points,
      });
    });
    Object.entries(existing.thresholds || {}).forEach(([key, threshold]) => {
      thresholds[key] = cloneJson(threshold);
    });
  }

  normalizedCharts.forEach((incoming) => {
    const current = chartMap.get(incoming.key) || {
      key: incoming.key,
      name: incoming.name,
      unit: incoming.unit,
      points: new Map(),
    };
    current.name = incoming.name || current.name;
    current.unit = incoming.unit || current.unit;
    incoming.points.forEach((value, timestamp) => current.points.set(timestamp, value));
    chartMap.set(incoming.key, current);
    thresholds[incoming.key] = {
      name: current.name,
      unit: current.unit,
      min: incoming.threshold.min,
      max: incoming.threshold.max,
    };
  });

  const trendCharts = [...chartMap.values()]
    .sort((left, right) => left.key.localeCompare(right.key))
    .map((chart) => {
      const points = [...chart.points.entries()].sort((left, right) => (
        historyTimestampValue(left[0]) - historyTimestampValue(right[0])
      ));
      return {
        key: chart.key,
        name: chart.name,
        unit: chart.unit,
        axisDataX: points.map(([timestamp]) => timestamp),
        seriesData: points.map(([, value]) => value),
      };
    });

  return {
    schemaVersion: "telemetry-history-1.0",
    satelliteId: String(satelliteId),
    updatedAt: new Date().toISOString(),
    thresholds,
    trendCharts,
    anomalyEvents: buildAnomalyEvents(trendCharts, thresholds),
  };
}

function emptyTelemetryHistory(satelliteId) {
  return {
    schemaVersion: "telemetry-history-1.0",
    satelliteId: String(satelliteId),
    updatedAt: null,
    thresholds: {},
    trendCharts: [],
    anomalyEvents: [],
  };
}

function sendJson(response, status, payload, extraHeaders = {}) {
  const body = Buffer.from(JSON.stringify(payload), "utf8");
  response.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": body.length,
    "Cache-Control": "no-store",
    ...extraHeaders,
  });
  response.end(body);
}

function sendEmpty(response, status, extraHeaders = {}) {
  response.writeHead(status, { "Cache-Control": "no-store", ...extraHeaders });
  response.end();
}

function readJsonBody(request) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let bytes = 0;
    request.on("data", (chunk) => {
      bytes += chunk.length;
      if (bytes > MAX_JSON_BYTES) {
        reject(new HttpError(413, "请求体过大", "BODY_TOO_LARGE"));
        request.destroy();
        return;
      }
      chunks.push(chunk);
    });
    request.on("end", () => {
      try {
        const text = Buffer.concat(chunks).toString("utf8");
        resolve(text ? JSON.parse(text) : {});
      } catch {
        reject(new HttpError(400, "JSON格式无效", "INVALID_JSON"));
      }
    });
    request.on("error", reject);
  });
}

function validateTelemetry(telemetry) {
  if (!telemetry || typeof telemetry !== "object" || Array.isArray(telemetry)) {
    throw new HttpError(400, "遥测数据必须是对象", "TELEMETRY_INVALID");
  }
  const coreParameters = Array.isArray(telemetry.coreParameters) ? telemetry.coreParameters : [];
  const trendCharts = Array.isArray(telemetry.trendCharts) ? telemetry.trendCharts : [];
  if (!coreParameters.length && !trendCharts.length) {
    throw new HttpError(400, "遥测数据缺少核心参数或时间序列", "TELEMETRY_EMPTY");
  }
  if (coreParameters.length > 512 || trendCharts.length > 512) {
    throw new HttpError(400, "遥测通道数量超过限制", "TOO_MANY_CHANNELS");
  }
  trendCharts.forEach((chart) => {
    if (Array.isArray(chart && chart.seriesData) && chart.seriesData.length > 20000) {
      throw new HttpError(400, "单通道采样点超过20000", "TOO_MANY_SAMPLES");
    }
  });
}

function healthIndicatorFromAssessment(assessment) {
  return Object.fromEntries(
    (Array.isArray(assessment.units) ? assessment.units : []).map((unit) => {
      const key = String(unit.key || unit.name);
      const number = Number.isFinite(Number(unit.health)) ? Number(unit.health) : 0;
      const abnormal = unit.state !== "健康";
      return [key, {
        title: String(unit.name || key),
        status: abnormal ? 1 : 0,
        number: Math.max(0, Math.min(100, number)),
      }];
    }),
  );
}

function buildDataPackage(satelliteId, satelliteName, telemetry, assessment) {
  const dataPackage = cloneJson(telemetry);
  delete dataPackage.orbitParameters;
  dataPackage.satelliteId = String(satelliteId);
  dataPackage.satelliteName = String(satelliteName || telemetry.satelliteName || satelliteId);
  dataPackage.algorithmVersion = assessment.algorithmVersion;
  dataPackage.healthAssessment = assessment;
  dataPackage.subsystemHealth = assessment.subsystemHealth;
  dataPackage.detailData = {
    ...(dataPackage.detailData || {}),
    healthIndicator: {
      ...((dataPackage.detailData || {}).healthIndicator || {}),
      ...healthIndicatorFromAssessment(assessment),
    },
    totalHealthInfo: assessment.totalHealthInfo,
    childHealthInfo: assessment.childHealthInfo,
    healthAssessment: assessment,
  };
  return dataPackage;
}

function clamp01(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.min(1, Math.max(0, number)) : fallback;
}

function normalizeLayout(body, satelliteId) {
  const cards = Array.isArray(body.cards) ? body.cards : [];
  const connections = Array.isArray(body.connections) ? body.connections : [];
  if (cards.length > 64) throw new HttpError(400, "指标卡片超过64个", "TOO_MANY_CARDS");
  if (connections.length > 128) throw new HttpError(400, "连线超过128条", "TOO_MANY_CONNECTIONS");
  const normalizedCards = cards.map((card, index) => {
    const metricId = String(card.metricId || "").trim();
    if (!metricId) throw new HttpError(400, `第${index + 1}张卡片缺少指标ID`, "METRIC_ID_REQUIRED");
    const width = Math.max(0.10, Math.min(0.45, clamp01(card.width, 0.18)));
    const height = Math.max(0.08, Math.min(0.32, clamp01(card.height, 0.12)));
    return {
      id: String(card.id || `card-${index + 1}`).slice(0, 100),
      metricId: metricId.slice(0, 200),
      metricName: String(card.metricName || metricId).slice(0, 200),
      x: Math.min(1 - width, clamp01(card.x, 0.05)),
      y: Math.min(1 - height, clamp01(card.y, 0.08 + index * 0.13)),
      width,
      height,
    };
  });
  const cardIds = new Set(normalizedCards.map((card) => card.id));
  const normalizedConnections = connections
    .filter((connection) => cardIds.has(String(connection.cardId || "")))
    .map((connection) => ({
      cardId: String(connection.cardId),
      cardAnchor: ["left", "right", "top", "bottom"].includes(connection.cardAnchor)
        ? connection.cardAnchor
        : "right",
      componentName: String(connection.componentName || "卫星部件").slice(0, 100),
      targetX: clamp01(connection.targetX, 0.5),
      targetY: clamp01(connection.targetY, 0.5),
    }));
  return {
    schemaVersion: "popup-layout-1.0",
    satelliteId: String(satelliteId),
    cards: normalizedCards,
    connections: normalizedConnections,
    updatedAt: new Date().toISOString(),
  };
}

function decodeImage(body) {
  const match = String(body.imageDataUrl || "").match(/^data:(image\/(?:png|jpeg));base64,([A-Za-z0-9+/=\r\n]+)$/);
  if (!match) throw new HttpError(400, "仅支持PNG或JPEG的Data URL", "IMAGE_INVALID");
  const buffer = Buffer.from(match[2].replace(/\s/g, ""), "base64");
  if (!buffer.length || buffer.length > MAX_IMAGE_BYTES) {
    throw new HttpError(413, "图片为空或超过12MB", "IMAGE_TOO_LARGE");
  }
  return { mime: match[1], extension: match[1] === "image/png" ? "png" : "jpg", buffer };
}

function matchSatelliteRoute(pathname, suffix) {
  const pattern = new RegExp(`^/api/satellites/([^/]+)/${suffix}$`);
  const match = pathname.match(pattern);
  return match ? decodeURIComponent(match[1]) : null;
}

function createHealthServer(options = {}) {
  const storageRoot = path.resolve(options.storageRoot || DEFAULT_STORAGE);
  ensureDirectories(storageRoot);
  const restoredStates = new Set();

  function ensureStateRestored(satelliteId, key) {
    if (restoredStates.has(satelliteId)) return;
    const snapshot = readJsonFile(path.join(storageRoot, "state", `${key}.json`));
    if (snapshot && snapshot.modelSchemaVersion === algorithm.modelInfo().schemaVersion) {
      algorithm.restoreState(satelliteId, snapshot);
    }
    restoredStates.add(satelliteId);
  }

  function clearAllPersistedStates() {
    const stateDir = path.join(storageRoot, "state");
    fs.readdirSync(stateDir).filter((name) => name.endsWith(".json")).forEach((name) => removeIfExists(path.join(stateDir, name)));
    restoredStates.clear();
  }

  return http.createServer(async (request, response) => {
    const startedAt = Date.now();
    const requestUrl = new URL(request.url || "/", "http://127.0.0.1");
    const pathname = requestUrl.pathname.replace(/\/+$/, "") || "/";
    const corsHeaders = {
      "Access-Control-Allow-Origin": request.headers.origin || "*",
      "Access-Control-Allow-Headers": "Content-Type",
      "Access-Control-Allow-Methods": "GET,POST,PUT,OPTIONS",
    };

    try {
      if (request.method === "OPTIONS") {
        sendEmpty(response, 204, corsHeaders);
        return;
      }
      if (request.method === "GET" && pathname === "/health") {
        sendJson(response, 200, {
          ok: true,
          service: "satellite-health-v2",
          algorithmVersion: algorithm.VERSION,
          runtimeMode: RUNTIME_MODE,
          algorithmWindowSize: ALGORITHM_WINDOW_SIZE,
          model: algorithm.modelInfo(),
          stateEntries: algorithm.stateCount(),
          uptimeSeconds: Math.round(process.uptime()),
        }, corsHeaders);
        return;
      }
      if (request.method === "GET" && pathname === "/api/health/runtime") {
        sendJson(response, 200, {
          ok: true,
          data: {
            runtimeMode: RUNTIME_MODE,
            algorithmWindowSize: ALGORITHM_WINDOW_SIZE,
            statefulDbn: true,
            persistentDbnState: true,
            acceptedAnalyzeInputs: ["single-frame", "legacy-window"],
            trainingIngestEnabled: RUNTIME_MODE === "training" || RUNTIME_MODE === "hybrid",
            model: algorithm.modelInfo(),
            stateEntries: algorithm.stateCount(),
          },
        }, corsHeaders);
        return;
      }
      if (request.method === "GET" && pathname === "/api/health/model") {
        sendJson(response, 200, { ok: true, data: algorithm.modelInfo() }, corsHeaders);
        return;
      }
      if (request.method === "POST" && pathname === "/api/health/model/reload") {
        const model = algorithm.reloadModel();
        clearAllPersistedStates();
        sendJson(response, 200, { ok: true, data: { model, dbnStateReset: true } }, corsHeaders);
        return;
      }
      if (request.method === "POST" && pathname === "/api/health/training/ingest") {
        if (!(RUNTIME_MODE === "training" || RUNTIME_MODE === "hybrid")) {
          throw new HttpError(409, "当前服务不是 training/hybrid 模式", "TRAINING_MODE_DISABLED");
        }
        const body = await readJsonBody(request);
        const satelliteId = String(body.satelliteId || "").trim();
        satelliteKey(satelliteId);
        const telemetry = body.telemetry || body.dataPackage;
        validateTelemetry(telemetry);
        if (!body.groundTruth || ![0, 1].includes(Number(body.groundTruth.label))) {
          throw new HttpError(400, "训练采集必须携带 groundTruth.label=0/1", "GROUND_TRUTH_REQUIRED");
        }
        const trainingRecord = {
          schemaVersion: "telemetry-training-frame-1.0",
          receivedAt: new Date().toISOString(),
          satelliteId,
          satelliteName: String(body.satelliteName || satelliteId),
          scenarioId: body.scenarioId || body.groundTruth.scenarioId || null,
          timestamp: body.timestamp || body.groundTruth.timestamp || telemetry.timestamp || null,
          telemetry,
          groundTruth: body.groundTruth,
        };
        appendJsonLine(path.join(storageRoot, "training", "frames.ndjson"), trainingRecord);
        sendJson(response, 202, { ok: true, data: { accepted: true, label: Number(body.groundTruth.label), scenarioId: trainingRecord.scenarioId } }, corsHeaders);
        return;
      }
      if (request.method === "POST" && pathname === "/api/health/analyze") {
        if (RUNTIME_MODE === "training") {
          throw new HttpError(409, "training 模式不执行在线健康评估；请使用 /api/health/training/ingest", "ONLINE_MODE_DISABLED");
        }
        const body = await readJsonBody(request);
        const satelliteId = String(body.satelliteId || "").trim();
        const key = satelliteKey(satelliteId);
        const telemetry = cloneJson(body.telemetry || body.dataPackage);
        if (telemetry && !telemetry.timestamp) telemetry.timestamp = body.timestamp || body.groundTruth?.timestamp || new Date().toISOString();
        validateTelemetry(telemetry);
        const inputFormat = (Array.isArray(telemetry.trendCharts) && telemetry.trendCharts.some((chart) => Array.isArray(chart?.seriesData) && chart.seriesData.length > 1))
          ? "legacy-window"
          : "single-frame";
        const historyPath = path.join(storageRoot, "history", `${key}.json`);
        const history = mergeTelemetryHistory(readJsonFile(historyPath), satelliteId, telemetry);
        const algorithmTelemetry = buildAlgorithmTelemetry(telemetry, history, ALGORITHM_WINDOW_SIZE);
        ensureStateRestored(satelliteId, key);
        const assessment = algorithm.calculate({ telemetry: algorithmTelemetry, satelliteId });
        const dataPackage = buildDataPackage(satelliteId, body.satelliteName, algorithmTelemetry, assessment);
        const record = {
          schemaVersion: "health-result-2.0",
          satelliteId,
          satelliteName: dataPackage.satelliteName,
          calculatedAt: new Date().toISOString(),
          calculationMs: Date.now() - startedAt,
          algorithmVersion: algorithm.VERSION,
          runtime: {
            mode: RUNTIME_MODE,
            inputFormat,
            algorithmWindowSize: ALGORITHM_WINDOW_SIZE,
            availableHistoryPoints: latestHistoryPointCount(history),
            statefulDbn: true,
            persistentDbnState: true,
            model: algorithm.modelInfo(),
          },
          assessment,
          dataPackage,
        };
        atomicWriteJson(path.join(storageRoot, "health", `${key}.json`), record);
        atomicWriteJson(historyPath, history);
        atomicWriteJson(path.join(storageRoot, "state", `${key}.json`), algorithm.snapshotState(satelliteId));
        sendJson(response, 200, { ok: true, data: record }, corsHeaders);
        return;
      }

      const resetId = matchSatelliteRoute(pathname, "health/reset");
      if (request.method === "POST" && resetId !== null) {
        const body = await readJsonBody(request);
        const key = satelliteKey(resetId);
        algorithm.reset(resetId);
        restoredStates.delete(String(resetId));
        removeIfExists(path.join(storageRoot, "state", `${key}.json`));
        if (body.clearHistory === true) {
          removeIfExists(path.join(storageRoot, "history", `${key}.json`));
          removeIfExists(path.join(storageRoot, "health", `${key}.json`));
        }
        sendJson(response, 200, { ok: true, data: { satelliteId: String(resetId), dbnStateReset: true, historyCleared: body.clearHistory === true } }, corsHeaders);
        return;
      }

      const healthId = matchSatelliteRoute(pathname, "health/latest");
      if (request.method === "GET" && healthId !== null) {
        const record = readJsonFile(path.join(storageRoot, "health", `${satelliteKey(healthId)}.json`));
        if (!record) throw new HttpError(404, "尚无健康计算结果", "HEALTH_NOT_FOUND");
        sendJson(response, 200, { ok: true, data: record }, corsHeaders);
        return;
      }

      const historyId = matchSatelliteRoute(pathname, "telemetry-history");
      if (request.method === "GET" && historyId !== null) {
        const history = readJsonFile(
          path.join(storageRoot, "history", `${satelliteKey(historyId)}.json`),
        );
        sendJson(response, 200, {
          ok: true,
          data: history || emptyTelemetryHistory(historyId),
        }, corsHeaders);
        return;
      }

      const layoutId = matchSatelliteRoute(pathname, "popup-layout");
      if (layoutId !== null && request.method === "GET") {
        const layout = readJsonFile(path.join(storageRoot, "layouts", `${satelliteKey(layoutId)}.json`));
        sendJson(response, 200, {
          ok: true,
          data: layout || normalizeLayout({ cards: [], connections: [] }, layoutId),
        }, corsHeaders);
        return;
      }
      if (layoutId !== null && request.method === "PUT") {
        const body = await readJsonBody(request);
        const layout = normalizeLayout(body, layoutId);
        atomicWriteJson(path.join(storageRoot, "layouts", `${satelliteKey(layoutId)}.json`), layout);
        sendJson(response, 200, { ok: true, data: layout }, corsHeaders);
        return;
      }

      const imageId = matchSatelliteRoute(pathname, "front-view");
      if (imageId !== null && request.method === "POST") {
        const body = await readJsonBody(request);
        const key = satelliteKey(imageId);
        const image = decodeImage(body);
        const fileName = `${key}.${image.extension}`;
        const filePath = path.join(storageRoot, "front-views", fileName);
        fs.writeFileSync(filePath, image.buffer);
        const metadata = {
          schemaVersion: "front-view-1.0",
          satelliteId: String(imageId),
          mime: image.mime,
          fileName,
          bytes: image.buffer.length,
          width: Number(body.width) || null,
          height: Number(body.height) || null,
          capturedAt: new Date().toISOString(),
        };
        atomicWriteJson(path.join(storageRoot, "front-views", `${key}.json`), metadata);
        sendJson(response, 200, {
          ok: true,
          data: { ...metadata, url: `/api/satellites/${encodeURIComponent(imageId)}/front-view` },
        }, corsHeaders);
        return;
      }
      if (imageId !== null && request.method === "GET") {
        const key = satelliteKey(imageId);
        const metadata = readJsonFile(path.join(storageRoot, "front-views", `${key}.json`));
        if (!metadata) throw new HttpError(404, "尚无卫星正视图", "FRONT_VIEW_NOT_FOUND");
        if (requestUrl.searchParams.get("meta") === "1") {
          sendJson(response, 200, { ok: true, data: metadata }, corsHeaders);
          return;
        }
        const imagePath = path.join(storageRoot, "front-views", path.basename(metadata.fileName));
        const stat = fs.statSync(imagePath);
        response.writeHead(200, {
          "Content-Type": metadata.mime,
          "Content-Length": stat.size,
          "Cache-Control": "no-cache",
          ...corsHeaders,
        });
        fs.createReadStream(imagePath).pipe(response);
        return;
      }

      throw new HttpError(404, "接口不存在", "NOT_FOUND");
    } catch (error) {
      const status = error instanceof HttpError ? error.status : 500;
      const code = error instanceof HttpError ? error.code : "INTERNAL_ERROR";
      if (status >= 500) console.error("[health-v2] request failed", error);
      if (!response.headersSent) {
        sendJson(response, status, { ok: false, error: { code, message: error.message || "服务异常" } }, corsHeaders);
      } else {
        response.destroy();
      }
    }
  });
}

function start() {
  const server = createHealthServer();
  server.listen(DEFAULT_PORT, DEFAULT_HOST, () => {
    console.log(JSON.stringify({
      service: "satellite-health-v2",
      host: DEFAULT_HOST,
      port: DEFAULT_PORT,
      algorithmVersion: algorithm.VERSION,
      runtimeMode: RUNTIME_MODE,
      algorithmWindowSize: ALGORITHM_WINDOW_SIZE,
      model: algorithm.modelInfo(),
      storage: DEFAULT_STORAGE,
    }));
  });
}

if (require.main === module) start();

module.exports = {
  createHealthServer,
  buildDataPackage,
  normalizeLayout,
  normalizeHistoryTimestamp,
  mergeTelemetryHistory,
  buildAlgorithmTelemetry,
  buildAnomalyEvents,
};
