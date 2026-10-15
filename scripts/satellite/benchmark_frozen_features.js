"use strict";
/**
 * V6 原版特征提取器的独立计时。
 * 冻结特征来自原始 JSON，计时结果写入 results/reproduced 防止覆盖历史记录。
 * 统计采用每次运行的微秒值，包含 p95 与 p99，禁止人为校准到论文的59.8%。
 */
const fs=require('node:fs'),path=require('node:path'),os=require('node:os');
const {performance}=require('node:perf_hooks');
const ROOT=path.resolve(__dirname,'../..');
const src=path.join(ROOT,'src','satellite');
const {TelemetrySimulator}=require(path.join(src,'satellite-datastream-service/telemetry-simulator.js'));
const {extractFeatureVector}=require(path.join(src,'health-v2-service/core/health-features.js'));
const model=JSON.parse(fs.readFileSync(path.join(src,'health-v2-service/core/models/subsystem-health-model-v6.json'),'utf8'));
const config=JSON.parse(fs.readFileSync(path.join(ROOT,'configs/benchmark.json'),'utf8'));
const args=process.argv.slice(2);
function arg(name,defaultValue){const i=args.indexOf(name);return i>=0?args[i+1]:defaultValue;}
const repeats=Number(arg('--repeats',config.repeats)),warmup=Number(arg('--warmup',config.warmup));
if(!Number.isInteger(repeats)||repeats<10||!Number.isInteger(warmup)||warmup<0) throw Error('repeats必须>=10；warmup必须>=0');
const outdir=path.resolve(arg('--out',path.join(ROOT,'results/reproduced/satellite/deployment_benchmark')));
const selected=[...new Set(Object.values(model.subsystems).flatMap(x=>x.selectedFeatures.map(f=>f.name)))];
if(selected.length!==140)throw Error(`冻结数量不合法: ${selected.length}`);
const sim=new TelemetrySimulator({seed:424242,windowSize:64,scenarioId:'benchmark'});
let packet;for(let i=0;i<80;i++)packet=sim.nextPoint(Date.UTC(2026,0,1));const telemetry=packet.telemetry;
const rows=[];
function bench(name,fn){
  for(let i=0;i<warmup;i++) fn();
  const times=[];
  for(let i=0;i<repeats;i++){
    const start=performance.now(); fn(); const elapsed=(performance.now()-start)*1000;
    times.push(elapsed); rows.push(`${name},${i+1},${elapsed}`);
  }
  const s=times.slice().sort((a,b)=>a-b);
  const quantile=q=>s[Math.min(s.length-1,Math.floor(q*s.length))];
  return {n:times.length,warmup,meanUs:times.reduce((a,b)=>a+b,0)/times.length,medianUs:quantile(0.5),p95Us:quantile(0.95),p99Us:quantile(0.99),minUs:s[0],maxUs:s.at(-1)};
}
const full=bench('full_582',()=>extractFeatureVector(telemetry));
const frozen=bench('frozen_140',()=>extractFeatureVector(telemetry,{selectedFeatureNames:selected,scales:[8,24,64]}));
const summary={schema:'MSF-MOFS-V6-original-frozen-benchmark-1',environment:{node:process.version,platform:os.platform(),release:os.release(),architecture:os.arch(),cpus:[...new Set(os.cpus().map(x=>x.model))],cpuCount:os.cpus().length},candidate:582,selected:140,full,frozen,medianReduction:1-frozen.medianUs/full.medianUs,medianSpeedup:full.medianUs/frozen.medianUs,paperHistoricalMedianReduction:0.598,note:'论文的59.8%来自不同运行环境，不要求本机吻合'};
fs.mkdirSync(outdir,{recursive:true});
fs.writeFileSync(path.join(outdir,'raw_timings.csv'),'method,iteration,elapsed_us\n'+rows.join('\n')+'\n');
fs.writeFileSync(path.join(outdir,'summary.json'),JSON.stringify(summary,null,2)+'\n');
console.log(JSON.stringify(summary,null,2));
