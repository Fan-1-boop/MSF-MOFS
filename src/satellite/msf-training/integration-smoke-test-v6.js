"use strict";

const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { createHealthServer } = require("../health-v2-service/service/server.js");
const algorithm = require("../health-v2-service/core/health-algorithm.js");
const { TelemetrySimulator } = require("../satellite-datastream-service/telemetry-simulator.js");
const { postJson, toSingleFrame } = require("../satellite-datastream-service/stream-service.js");

async function listen(server) {
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  return server.address().port;
}

async function main() {
  const storageRoot = fs.mkdtempSync(path.join(os.tmpdir(), "msf-health-v4-"));
  const satelliteId = "SMOKE-V6";
  const simulator = new TelemetrySimulator({
    seed: 20260826,
    fault: "param_4:drift:0.006:70:180",
    scenarioId: "smoke-v6",
    windowSize: 64,
    stepMs: 1000,
  });
  simulator.satelliteId = satelliteId;
  simulator.satelliteName = "V6 Smoke";

  let server = createHealthServer({ storageRoot });
  let port = await listen(server);
  let last;
  for (let i = 0; i < 90; i += 1) {
    const packet = simulator.nextPoint(Date.UTC(2026, 0, 1));
    const response = await postJson(`http://127.0.0.1:${port}/api/health/analyze`, toSingleFrame(packet, false));
    last = response.body.data;
  }
  if (last.runtime.inputFormat !== "single-frame") throw new Error("single-frame input was not recognized");
  if (last.assessment.msfV6.windowReadiness !== 1) throw new Error("64-point V6 feature window did not become ready");
  if (last.runtime.availableHistoryPoints !== 90) throw new Error(`unexpected persisted history count: ${last.runtime.availableHistoryPoints}`);
  await new Promise((resolve) => server.close(resolve));

  // Simulate process memory loss while keeping disk state/history.
  algorithm.reset(satelliteId);
  server = createHealthServer({ storageRoot });
  port = await listen(server);
  const packet = simulator.nextPoint(Date.UTC(2026, 0, 1));
  const response = await postJson(`http://127.0.0.1:${port}/api/health/analyze`, toSingleFrame(packet, false));
  if (response.body.data.runtime.availableHistoryPoints !== 91) throw new Error("persisted history was not reused after restart");
  if (algorithm.stateCount() < 1) throw new Error("DBN state was not restored/recreated after restart");
  await new Promise((resolve) => server.close(resolve));

  const summary = {
    ok: true,
    storageRoot,
    finalHealth: response.body.data.assessment.overallScore,
    historyPoints: response.body.data.runtime.availableHistoryPoints,
    selectedFeatureCount: response.body.data.assessment.msfV6.selectedFeatureCountUnion,
    model: algorithm.modelInfo(),
  };
  console.log(JSON.stringify(summary, null, 2));
  fs.rmSync(storageRoot, { recursive: true, force: true });
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
