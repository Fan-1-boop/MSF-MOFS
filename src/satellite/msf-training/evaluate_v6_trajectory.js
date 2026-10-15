"use strict";
const fs=require('node:fs');const path=require('node:path');
const algorithm=require('../health-v2-service/core/health-algorithm.js');
const {TelemetrySimulator,PARAMETERS}=require('../satellite-datastream-service/telemetry-simulator.js');
function mean(a){return a.length?a.reduce((x,y)=>x+y,0)/a.length:null;}
function targetSubsystems(key){const u=algorithm.definitions.unitDefinitions.find(x=>x.parameters.includes(key));return u?Object.keys(u.subsystems):[];}
function run(parameter,index){
 const id=`v6-traj-${parameter.key}`;algorithm.reset(id);const startTick=80,length=320;const fault={key:parameter.key,mode:'drift',magnitude:0,magnitudeStart:0.0001,magnitudeEnd:0.012,startTick,endTick:length+1};
 const sim=new TelemetrySimulator({seed:880000+index*101,faults:[fault],windowSize:64,scenarioId:id});const targets=targetSubsystems(parameter.key);let firstWarning=null,firstLimit=null;const rows=[];
 for(let t=0;t<length;t++){
  const packet=sim.nextPoint(Date.UTC(2026,0,1));const a=algorithm.calculate({telemetry:packet.telemetry,satelliteId:id});const subs=a.subsystems.filter(s=>targets.includes(s.id));const ev=subs.length?Math.max(...subs.map(s=>Number(s.evidence||0))):0;const sh=subs.length?Math.min(...subs.map(s=>Number(s.score||100))):a.overallScore;
  const cur=packet.telemetry.coreParameters.find(x=>x.key===parameter.key);const th=packet.telemetry.thresholds[parameter.key];const violated=cur.value<th.min||cur.value>th.max;const tf=packet.groundTruth.faults.find(x=>x.key===parameter.key);const ne=tf?Math.abs(tf.normalizedEffect||0):0;
  let stage='normal';if(t>=startTick){if(violated)stage='fault';else if(ne>=.04)stage='degraded';else stage='transition';}
  if(t>=startTick && firstWarning===null && (ev>=.5||sh<85))firstWarning=t;if(t>=startTick&&violated&&firstLimit===null)firstLimit=t;
  rows.push({t,stage,overall:a.overallScore,subsystemHealth:sh,evidence:ev,limit:violated?1:0,normalizedEffect:ne});
 }
 const ss={};for(const st of ['normal','transition','degraded','fault']){const p=rows.filter(r=>r.t>=64&&r.stage===st);ss[st]={points:p.length,meanOverall:mean(p.map(r=>r.overall)),meanSubsystem:mean(p.map(r=>r.subsystemHealth)),meanEvidence:mean(p.map(r=>r.evidence))}}
 return {parameter:parameter.key,targets,firstWarning,firstLimit,warningLatency:firstWarning===null?null:firstWarning-startTick,leadToLimit:firstWarning!==null&&firstLimit!==null?firstLimit-firstWarning:null,early:firstWarning!==null&&firstLimit!==null&&firstWarning<firstLimit,stages:ss};
}
function main(){const runs=PARAMETERS.map(run);const lead=runs.filter(r=>r.leadToLimit!==null).map(r=>r.leadToLimit);const stageNames=['normal','transition','degraded','fault'];const stageMean={};for(const st of stageNames){const rr=runs.map(r=>r.stages[st]).filter(x=>x.points>0);stageMean[st]={meanOverall:mean(rr.map(x=>x.meanOverall)),meanSubsystem:mean(rr.map(x=>x.meanSubsystem)),meanEvidence:mean(rr.map(x=>x.meanEvidence))}}
 const s={schemaVersion:'msf-v6-trajectory-1.0',generatedAt:new Date().toISOString(),scenarios:runs.length,warnings:runs.filter(r=>r.firstWarning!==null).length,limitCrossings:runs.filter(r=>r.firstLimit!==null).length,earlyWarnings:runs.filter(r=>r.early).length,meanWarningLatency:mean(runs.filter(r=>r.warningLatency!==null).map(r=>r.warningLatency)),meanLeadToLimit:mean(lead),stageMean,runs};const out=path.join(__dirname,'experiments','v6_final','trajectory_evaluation.json');fs.writeFileSync(out,JSON.stringify(s,null,2)+'\n');console.log(JSON.stringify({...s,runs:undefined},null,2));}
if(require.main===module)main();
