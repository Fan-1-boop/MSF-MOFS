// 中文说明：本文件仅适配原 V8 确认脚本的源码相对路径和输出目录；算法逻辑保持原样。
"use strict";
const fs=require('node:fs'), path=require('node:path');
const ROOT=path.resolve(__dirname,'../../results/reproduced/satellite/v8_confirmation');const V6=path.resolve(__dirname,'../../src/satellite');fs.mkdirSync(path.join(ROOT,'experiments'),{recursive:true});
const algorithm=require(path.join(V6,'health-v2-service','core','health-algorithm.js'));
const {TelemetrySimulator,PARAMETERS,FAULT_MODES,COUPLED_FAULT_MODES}=require(path.join(V6,'satellite-datastream-service','telemetry-simulator.js'));
const {RELATION_PAIRS}=require(path.join(V6,'health-v2-service','core','health-features.js'));
const {UNI_MAGNITUDE}=require(path.join(V6,'satellite-datastream-service','generate-training-data-v6.js'));
const CRITICAL=new Set(['p01_p02','p03_p04','p06_p12','p10_p11','p01_p08']);
const PAIRS=RELATION_PAIRS.filter(p=>CRITICAL.has(p.id));
const SEVERITIES=[0.55,0.85,1.15], LAGS=[6,10,14];
const pairSubsystem={p01_p02:'power',p03_p04:'thermal',p06_p12:'attitude',p10_p11:'communication',p01_p08:'power'};
function mean(a){return a.length?a.reduce((x,y)=>x+y,0)/a.length:null;}
function nuisance(seed){return {noiseScale:0.75+(seed%17)/34,amplitudeScale:0.78+(seed%13)/30};}
function onset(seed){return 88+(seed%17);}
function uniMagnitude(mode,seed){return UNI_MAGNITUDE[mode]*(0.88+(seed%7)*0.045);}
function targetSubsystemsForParam(key){const unit=algorithm.definitions.unitDefinitions.find(u=>u.parameters.includes(key));return unit?Object.keys(unit.subsystems):[];}
function run({id,seed,fault=null,coupledFault=null,length=256,startTick=96}){
  algorithm.reset(id); const nz=nuisance(seed);
  const sim=new TelemetrySimulator({seed,faults:fault?[fault]:[],coupledFaults:coupledFault?[coupledFault]:[],windowSize:64,stepMs:1000,scenarioId:id,...nz});
  const targets=fault?targetSubsystemsForParam(fault.key):coupledFault?[pairSubsystem[coupledFault.pairId]]:[];
  let first=null; const rows=[];
  for(let t=0;t<length;t++){
    const packet=sim.nextPoint(Date.UTC(2028,0,1));
    const a=algorithm.calculate({telemetry:packet.telemetry,satelliteId:id});
    const selected=a.subsystems.filter(s=>targets.includes(s.id));
    const targetEvidence=selected.length?Math.max(...selected.map(s=>Number(s.evidence||0))):0;
    const targetHealth=selected.length?Math.min(...selected.map(s=>Number(s.score||100))):100;
    const alarm=(targetEvidence>=0.5 || targetHealth<85 || a.overallScore<85);
    if(t>=startTick && alarm && first===null) first=t;
    rows.push({t,overall:a.overallScore,targetEvidence,targetHealth,alarm});
  }
  const evalRows=rows.filter(r=>r.t>=Math.max(64,startTick));
  return {id,type:coupledFault?'coupled':fault?'univariate':'normal',mode:coupledFault?.mode||fault?.mode||'normal',target:fault?.key||coupledFault?.pairId||'none',targets,detected:first!==null,latency:first===null?null:first-startTick,meanOverall:mean(evalRows.map(r=>r.overall)),minOverall:Math.min(...evalRows.map(r=>r.overall)),maxTargetEvidence:Math.max(...evalRows.map(r=>r.targetEvidence)),minTargetHealth:Math.min(...evalRows.map(r=>r.targetHealth)),alarmFraction:mean(evalRows.map(r=>r.alarm?1:0))};
}
function main(){
  const results=[],length=256;
  for(let i=0;i<30;i++){const seed=12600000+i*211;results.push(run({id:`v8-confirm-normal-${i}`,seed,length,startTick:96}));}
  PARAMETERS.forEach((p,pi)=>FAULT_MODES.forEach((mode,mi)=>{const seed=12700000+pi*1009+mi*83;const st=onset(seed);results.push(run({id:`v8-confirm-uni-${p.key}-${mode}`,seed,fault:{key:p.key,mode,magnitude:uniMagnitude(mode,seed),startTick:st,endTick:length+1},length,startTick:st}));}));
  PAIRS.forEach((pair,pi)=>COUPLED_FAULT_MODES.forEach((mode,mi)=>SEVERITIES.forEach((severity,si)=>{const seed=12800000+pi*5003+mi*419+si*97;const st=onset(seed);results.push(run({id:`v8-confirm-cpl-${pair.id}-${mode}-${si}`,seed,coupledFault:{pairId:pair.id,a:pair.a,b:pair.b,target:pair.b,group:pair.group,mode,severity,lag:LAGS[si],startTick:st,endTick:length+1},length,startTick:st}));})));
  const normal=results.filter(r=>r.type==='normal'),uni=results.filter(r=>r.type==='univariate'),cpl=results.filter(r=>r.type==='coupled');
  const bySubsystem={};for(const r of [...uni,...cpl])for(const s of r.targets){if(!bySubsystem[s])bySubsystem[s]=[];bySubsystem[s].push(r);}
  const summary={schemaVersion:'msf-v8-locked-health-confirmation-1.0',generatedAt:new Date().toISOString(),frozenModel:'V6 subsystem-health-model-v6.json',criterion:'target subsystem evidence >=0.5 OR target subsystem health <85 OR overall health <85',normal:{scenarios:normal.length,meanHealth:mean(normal.map(r=>r.meanOverall)),minHealth:Math.min(...normal.map(r=>r.minOverall)),falseAlarmScenarios:normal.filter(r=>r.detected).length},univariate:{scenarios:uni.length,detected:uni.filter(r=>r.detected).length,detectionRate:mean(uni.map(r=>r.detected?1:0)),meanLatency:mean(uni.filter(r=>r.latency!==null).map(r=>r.latency))},coupled:{scenarios:cpl.length,detected:cpl.filter(r=>r.detected).length,detectionRate:mean(cpl.map(r=>r.detected?1:0)),meanLatency:mean(cpl.filter(r=>r.latency!==null).map(r=>r.latency))},bySubsystem:Object.fromEntries(Object.entries(bySubsystem).map(([s,a])=>[s,{scenarios:a.length,detected:a.filter(r=>r.detected).length,detectionRate:mean(a.map(r=>r.detected?1:0)),meanLatency:mean(a.filter(r=>r.latency!==null).map(r=>r.latency))}])),scenarios:results};
  const out=path.join(ROOT,'experiments','health_confirmation.json');fs.writeFileSync(out,JSON.stringify(summary,null,2)+'\n');console.log(JSON.stringify({...summary,scenarios:undefined},null,2));
}
if(require.main===module)main();
