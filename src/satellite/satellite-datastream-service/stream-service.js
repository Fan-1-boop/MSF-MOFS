"use strict";

/**
 * 卫星遥测数据流服务：stream-service.js
 *
 * 作用：按可配置周期不断地产生遥测数据，并 POST 到现有健康算法服务
 *       （/api/health/analyze），复用其「按时间戳去重的增量历史合并」能力。
 *
 * 这样，无需改动健康服务与前端：
 *   - 后端历史文件持续增长；
 *   - 前端详情弹窗里的轮询定时器（默认 5 秒，可切 1/5/10/30/60 秒）
 *     会周期读取 /telemetry-history，绘制出逐点变新的曲线。
 *
 * 数据源抽象为可插拔接口（默认 TelemetrySimulator），
 * 以后接真实数据源时只需替换 dataSource，不动本文件的调度逻辑。
 *
 * 配置（环境变量，见 -h / --help）：
 *   STREAM_SATELLITE_ID   卫星ID（必需）
 *   STREAM_SATELLITE_NAME 卫星名称（默认 = 卫星ID）
 *   STREAM_HEALTH_API     健康服务 analyze 地址（默认 http://127.0.0.1:18098）
 *   STREAM_INTERVAL_MS    上传周期，毫秒（默认 5000）
 *   STREAM_POINTS_PER_TICK 每次上传追加的采样点数（默认 1）
 *   STREAM_STEP_MS        相邻采样点的时间步长，毫秒（默认 = interval/points）
 *   STREAM_SEED           模拟器随机种子（默认 20260814）
 *   STREAM_FAULT          故障注入，如 param_4:drift:0.03 / param_11:spike:20 / off
 *   STREAM_MODE           online | training（默认 online）
 *   STREAM_INPUT_FORMAT   frame | legacy-window（默认 frame）
 *   STREAM_INCLUDE_TRUTH  1=在线模式也附带 groundTruth（默认 0；training 自动携带）
 *   STREAM_DRY_RUN        1=只打印不真正发送
 */

const http = require("node:http");
const { URL } = require("node:url");
const { TelemetrySimulator } = require("./telemetry-simulator.js");

function envInt(name, fallback) {
  const value = Number(process.env[name]);
  return Number.isFinite(value) && value > 0 ? Math.round(value) : fallback;
}

function envStr(name, fallback) {
  const value = process.env[name];
  return typeof value === "string" && value.length ? value : fallback;
}

function config() {
  const satelliteId = envStr("STREAM_SATELLITE_ID", "");
  const satelliteName = envStr("STREAM_SATELLITE_NAME", satelliteId || "模拟星");
  const healthApi = envStr("STREAM_HEALTH_API", "http://127.0.0.1:18098").replace(/\/$/, "");
  const intervalMs = envInt("STREAM_INTERVAL_MS", 5000);
  const pointsPerTick = envInt("STREAM_POINTS_PER_TICK", 1);
  const stepMs = envInt("STREAM_STEP_MS", Math.max(1, Math.round(intervalMs / pointsPerTick)));
  const seed = Number(process.env.STREAM_SEED) || 20260814;
  const fault = envStr("STREAM_FAULT", "off");
  const dryRun = process.env.STREAM_DRY_RUN === "1";
  const windowSize = envInt("STREAM_WINDOW_SIZE", 64);
  const modeRaw = envStr("STREAM_MODE", "online").toLowerCase();
  const mode = ["online", "training"].includes(modeRaw) ? modeRaw : "online";
  const formatRaw = envStr("STREAM_INPUT_FORMAT", "frame").toLowerCase();
  const inputFormat = formatRaw === "legacy-window" ? "legacy-window" : "frame";
  const includeTruth = mode === "training" || process.env.STREAM_INCLUDE_TRUTH === "1";
  const scenarioId = envStr("STREAM_SCENARIO_ID", `${satelliteId || "sim"}-${seed}`);
  return { satelliteId, satelliteName, healthApi, intervalMs, pointsPerTick, stepMs, seed, fault, dryRun, windowSize, mode, inputFormat, includeTruth, scenarioId };
}

/** 用原生 http 发起 POST，避免外部依赖；成功后回调结果数据。 */
function postJson(url, payload, timeoutMs = 10000) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify(payload);
    const parsed = new URL(url);
    const request = http.request(
      {
        hostname: parsed.hostname,
        port: parsed.port || 80,
        path: `${parsed.pathname}${parsed.search || ""}`,
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(body),
        },
        timeout: timeoutMs,
      },
      (response) => {
        const chunks = [];
        response.on("data", (chunk) => chunks.push(chunk));
        response.on("end", () => {
          const text = Buffer.concat(chunks).toString("utf8");
          let parsedBody = null;
          try { parsedBody = text ? JSON.parse(text) : null; } catch { /* 忽略解析失败 */ }
          if (response.statusCode >= 200 && response.statusCode < 300) {
            resolve({ status: response.statusCode, body: parsedBody });
          } else {
            reject(new Error(`HTTP ${response.statusCode}：${parsedBody?.error?.message || text || "无响应体"}`));
          }
        });
      },
    );
    request.on("timeout", () => request.destroy(new Error("请求超时")));
    request.on("error", reject);
    request.end(body);
  });
}

function toSingleFrame(packet, includeTruth = false) {
  const telemetry = packet.telemetry || {};
  const timestamp = packet.groundTruth?.timestamp
    || telemetry.timestamp
    || telemetry.trendCharts?.[0]?.axisDataX?.slice(-1)[0]
    || new Date().toISOString();
  const frame = {
    schemaVersion: "telemetry-frame-1.0",
    satelliteId: packet.satelliteId,
    satelliteName: packet.satelliteName,
    scenarioId: packet.scenarioId || packet.groundTruth?.scenarioId || undefined,
    timestamp,
    telemetry: {
      timestamp,
      coreParameters: Array.isArray(telemetry.coreParameters) ? telemetry.coreParameters : [],
      thresholds: telemetry.thresholds || {},
    },
  };
  if (includeTruth && packet.groundTruth) frame.groundTruth = packet.groundTruth;
  return frame;
}

function buildOutgoingPacket(packet, cfg) {
  if (cfg.inputFormat === "legacy-window") {
    const legacy = { ...packet };
    if (!cfg.includeTruth) delete legacy.groundTruth;
    return legacy;
  }
  return toSingleFrame(packet, cfg.includeTruth);
}

/** 生成采样点并逐点 POST。默认只发送最新一帧，窗口由健康服务按卫星持久维护。 */
async function runTick(simulator, cfg, window, onResult) {
  for (let i = 0; i < cfg.pointsPerTick; i += 1) {
    const nowMs = Date.now();
    const packet = simulator.nextPoint(nowMs);
    packet.satelliteId = cfg.satelliteId;
    packet.satelliteName = cfg.satelliteName;
    const outgoing = buildOutgoingPacket(packet, cfg);

    if (cfg.dryRun) {
      const latest = packet.telemetry.trendCharts[0];
      onResult({ status: 0, dry: true, point: `${latest.axisDataX.slice(-1)[0]} ${latest.name}=${latest.seriesData.slice(-1)[0]}` });
      continue;
    }

    try {
      const endpoint = cfg.mode === "training" ? "/api/health/training/ingest" : "/api/health/analyze";
      const result = await postJson(`${cfg.healthApi}${endpoint}`, outgoing, 10000);
      const assessment = result.body?.data?.assessment || result.body?.assessment;
      const overall = cfg.mode === "training"
        ? "training-accepted"
        : (assessment?.overallScore ?? assessment?.totalHealthInfo?.score ?? assessment?.score ?? "?");
      onResult({ status: result.status, overall, point: new Date(nowMs).toISOString().slice(0, 19), mode: cfg.mode });
    } catch (error) {
      onResult({ status: -1, error: error.message });
    }
  }
}

function printHelp() {
  console.log(`
卫星遥测数据流服务（health-v2/datastream）

用法：
  node stream-service.js [--satellite <id>] [--interval <ms>] [--fault <spec>] [--dry-run]

环境变量（命令行参数覆盖同名环境变量）：
  STREAM_SATELLITE_ID    卫星ID（必需）
  STREAM_SATELLITE_NAME  卫星名称（默认 = 卫星ID）
  STREAM_HEALTH_API      健康服务 analyze 地址（默认 http://127.0.0.1:18098）
  STREAM_INTERVAL_MS     上传周期，毫秒（默认 5000）
  STREAM_POINTS_PER_TICK 每次上传追加采样点数（默认 1）
  STREAM_STEP_MS         相邻采样点时间步长，毫秒（默认 = interval/points）
  STREAM_SEED            模拟器随机种子
  STREAM_FAULT           故障注入：<key>:<mode>:<magnitude>[:startTick:endTick]，off=关闭
  STREAM_MODE            online | training（默认 online）
  STREAM_INPUT_FORMAT    frame | legacy-window（默认 frame）
  STREAM_INCLUDE_TRUTH   1=在线模式也附带 groundTruth（training 自动携带）
  STREAM_SCENARIO_ID     训练场景ID（默认 satelliteId-seed）
  STREAM_DRY_RUN         1=只打印不发送

故障注入示例：
  --fault param_4:drift:0.03   载荷温度渐变漂移越限（演示持续退化）
  --fault param_11:spike:20    下行速率周期性毛刺越限
  --fault off                  关闭
`);
}

function main() {
  const args = process.argv.slice(2);
  if (args.includes("-h") || args.includes("--help")) {
    printHelp();
    process.exit(0);
  }

  const cfg = config();

  function argValue(name) {
    const index = args.indexOf(name);
    return index >= 0 && index + 1 < args.length ? args[index + 1] : "";
  }
  const argSatellite = argValue("--satellite");
  const argInterval = argValue("--interval");
  const argFault = argValue("--fault");
  const argHealthApi = argValue("--health-api");
  const argMode = argValue("--mode");
  const argFormat = argValue("--format");
  const argScenario = argValue("--scenario");

  if (argSatellite) cfg.satelliteId = argSatellite;
  if (argInterval) cfg.intervalMs = Number(argInterval);
  if (argFault) cfg.fault = argFault;
  if (argHealthApi) cfg.healthApi = argHealthApi;
  if (["online", "training"].includes(argMode)) cfg.mode = argMode;
  if (["frame", "legacy-window"].includes(argFormat)) cfg.inputFormat = argFormat;
  if (argScenario) cfg.scenarioId = argScenario;
  if (cfg.mode === "training") cfg.includeTruth = true;
  if (args.includes("--include-truth")) cfg.includeTruth = true;
  if (args.includes("--dry-run")) cfg.dryRun = true;

  if (!cfg.satelliteId) {
    console.error("❌ 缺少卫星ID：请通过 --satellite <id> 或 STREAM_SATELLITE_ID 指定。");
    process.exit(2);
  }

  const simulator = new TelemetrySimulator({ seed: cfg.seed, fault: cfg.fault, windowSize: cfg.windowSize, stepMs: cfg.stepMs, scenarioId: cfg.scenarioId });
  simulator.satelliteId = cfg.satelliteId;
  simulator.satelliteName = cfg.satelliteName;

  console.log("———————————— 卫星遥测数据流服务 ————————————");
  console.log(`  卫星        ：${cfg.satelliteId}`);
  console.log(`  卫星名称    ：${cfg.satelliteName}`);
  console.log(`  健康服务    ：${cfg.healthApi}/api/health/analyze`);
  console.log(`  上传周期    ：${cfg.intervalMs} ms`);
  console.log(`  每次采样点数：${cfg.pointsPerTick}`);
  console.log(`  数据源      ：${simulator.describe()}`);
  console.log(`  Scenario    ：${cfg.scenarioId}`);
  console.log(`  服务模式    ：${cfg.mode}`);
  console.log(`  输入格式    ：${cfg.inputFormat}`);
  console.log(`  GroundTruth ：${cfg.includeTruth ? "携带" : "不发送"}`);
  console.log(`  运行模式    ：${cfg.dryRun ? "DRY RUN（不发送）" : "实时发送"}`);
  console.log("————————————————————————————————————————————");

  let successes = 0;
  let failures = 0;
  let lastOverall = null;

  const tick = () => runTick(simulator, cfg, null, (result) => {
    if (result.dry) {
      console.log(`[dry] ${result.point}`);
      return;
    }
    if (result.status >= 0) {
      successes += 1;
      lastOverall = result.overall;
      console.log(cfg.mode === "training"
        ? `✓ 训练帧已采集  已连续成功 ${successes} 次`
        : `✓ 上传成功 总体健康度=${lastOverall}  已连续成功 ${successes} 次`);
    } else {
      failures += 1;
      console.error(`✗ 上传失败（${result.error}） 连续失败 ${failures} 次`);
    }
  });

  tick();
  const timer = setInterval(tick, cfg.intervalMs);
  // 注意：不要调用 timer.unref()，否则定时器不会让事件循环保持存活，
  // 进程会在首次上传完成后立即退出，导致周期更新停止。

  const shutdown = () => {
    clearInterval(timer);
    console.log("\n已停止。统计 → 成功 " + successes + " 次，失败 " + failures + " 次。");
    process.exit(0);
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

if (require.main === module) main();

module.exports = { config, postJson, runTick, toSingleFrame, buildOutgoingPacket };
