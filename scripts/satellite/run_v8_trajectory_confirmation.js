// 中文说明：本文件仅适配原 V8 确认脚本的源码相对路径和输出目录；算法逻辑保持原样。
"use strict";
const fs=require('node:fs'),path=require('node:path');const ROOT=path.resolve(__dirname,'../../results/reproduced/satellite/v8_confirmation');const V6=path.resolve(__dirname,'../../src/satellite');fs.mkdirSync(path.join(ROOT,'experiments'),{recursive:true});
const algorithm=require(path.join(V6,'health-v2-service','core','health-algorithm.js'));
const {TelemetrySimulator,PARAMETERS}=require(path.join(V6,'satellite-datastream-service','telemetry-simulator.js'));
function mean(a){return a.length?a.reduce((x,y)=>x+y,0)/a.length:null;}
function nuisance(seed){return {noiseScale:0.75+(seed%17)/34,amplitudeScale:0.78+(seed%13)/30};}
function targetSubsystems(key){const u=algorithm.definitions.unitDefinitions.find(x=>x.parameters.includes(key));return u?Object.keys(u.subsystems):[];}
function run(parameter,index){
 const id=`v8-confirm-traj-${parameter.key}`,seed=13600000+index*509,startTick=88+(index%4)*4,length=360;
 const magnitudeStart=0.00008*(1+(index%3)*0.08),magnitudeEnd=0.0100+(index%5)*0.00045;
 algorithm.reset(id);const sim=new TelemetrySimulator({seed,faults:[{key:parameter.key,mode:'drift',magnitude:0,magnitudeStart,magnitudeEnd,startTick,endTick:length+1}],windowSize:64,scenarioId:id,...nuisance(seed)});const targets=targetSubsystems(parameter.key);let firstWarning=null,firstLimit=null;const rows=[];
 for(let t=0;t<length;t++){
  const packet=sim.nextPoint(Date.UTC(2028,0,1));const a=algorithm.calculate({telemetry:packet.telemetry,satelliteId:id});const subs=a.subsystems.filter(s=>targets.includes(s.id));const ev=subs.length?Math.max(...subs.map(s=>Number(s.evidence||0))):0;const sh=subs.length?Math.min(...subs.map(s=>Number(s.score||100))):a.overallScore;
  const cur=packet.telemetry.coreParameters.find(x=>x.key===parameter.key),th=packet.telemetry.thresholds[parameter.key];const violated=cur.value<th.min||cur.value>th.max;const tf=packet.groundTruth.faults.find(x=>x.key===parameter.key);const ne=tf?Math.abs(tf.normalizedEffect||0):0;
  let stage='normal';if(t>=startTick){if(violated)stage='fault';else if(ne>=.04)stage='degraded';else stage='transition';}
  if(t>=startTick&&firstWarning===null&&(ev>=.5||sh<85))firstWarning=t;if(t>=startTick&&violated&&firstLimit===null)firstLimit=t;
  rows.push({t,stage,overall:a.overallScore,subsystemHealth:sh,evidence:ev,limit:violated?1:0,normalizedEffect:ne});
 }
 const ss={};for(const st of ['normal','transition','degraded','fault']){const p=rows.filter(r=>r.t>=64&&r.stage===st);ss[st]={points:p.length,meanOverall:mean(p.map(r=>r.overall)),meanSubsystem:mean(p.map(r=>r.subsystemHealth)),meanEvidence:mean(p.map(r=>r.evidence))};}
 return {parameter:parameter.key,targets,startTick,magnitudeStart,magnitudeEnd,firstWarning,firstLimit,warningLatency:firstWarning===null?null:firstWarning-startTick,leadToLimit:firstWarning!==null&&firstLimit!==null?firstLimit-firstWarning:null,early:firstWarning!==null&&firstLimit!==null&&firstWarning<firstLimit,stages:ss};
}
function main(){const runs=PARAMETERS.map(run),lead=runs.filter(r=>r.leadToLimit!==null).map(r=>r.leadToLimit),stageMean={};for(const st of ['normal','transition','degraded','fault']){const rr=runs.map(r=>r.stages[st]).filter(x=>x.points>0);stageMean[st]={meanOverall:mean(rr.map(x=>x.meanOverall)),meanSubsystem:mean(rr.map(x=>x.meanSubsystem)),meanEvidence:mean(rr.map(x=>x.meanEvidence))};}
 const s={schemaVersion:'msf-v8-locked-trajectory-confirmation-1.0',generatedAt:new Date().toISOString(),scenarios:runs.length,warnings:runs.filter(r=>r.firstWarning!==null).length,limitCrossings:runs.filter(r=>r.firstLimit!==null).length,earlyWarnings:runs.filter(r=>r.early).length,meanWarningLatency:mean(runs.filter(r=>r.warningLatency!==null).map(r=>r.warningLatency)),meanLeadToLimit:mean(lead),stageMean,runs};fs.writeFileSync(path.join(ROOT,'experiments','trajectory_confirmation.json'),JSON.stringify(s,null,2)+'\n');console.log(JSON.stringify({...s,runs:undefined},null,2));}
if(require.main===module)main();
