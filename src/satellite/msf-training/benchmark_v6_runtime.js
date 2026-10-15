"use strict";
const fs=require('node:fs');const path=require('node:path');
const {performance}=require('node:perf_hooks');
const {TelemetrySimulator}=require('../satellite-datastream-service/telemetry-simulator.js');
const {extractFeatureVector}=require('../health-v2-service/core/health-features.js');
const algorithm=require('../health-v2-service/core/health-algorithm.js');
const model=JSON.parse(fs.readFileSync('../health-v2-service/core/models/subsystem-health-model-v6.json','utf8'));
const selected=[...new Set(Object.values(model.subsystems).flatMap(m=>m.selectedFeatures.map(f=>f.name)))];
const sim=new TelemetrySimulator({seed:424242,windowSize:64,scenarioId:'bench'});let packet;for(let i=0;i<80;i++)packet=sim.nextPoint(Date.UTC(2026,0,1));const tel=packet.telemetry;
function bench(fn,n=2000){for(let i=0;i<100;i++)fn();const a=[];for(let i=0;i<n;i++){const t=performance.now();fn();a.push((performance.now()-t)*1000)}a.sort((x,y)=>x-y);return {n,medianUs:a[Math.floor(n*.5)],p95Us:a[Math.floor(n*.95)],meanUs:a.reduce((x,y)=>x+y,0)/n};}
const all=bench(()=>extractFeatureVector(tel),2000);const sel=bench(()=>extractFeatureVector(tel,{selectedFeatureNames:selected,scales:[8,24,64]}),2000);algorithm.reset('bench');const e2e=bench(()=>algorithm.calculate({telemetry:tel,satelliteId:'bench'}),1000);
const out={schemaVersion:'msf-v6-runtime-1.0',candidateFeatures:582,selectedUnion:selected.length,featureReduction:1-selected.length/582,allFeatureExtraction:all,selectedFeatureExtraction:sel,extractionSpeedup:all.medianUs/sel.medianUs,extractionMedianReduction:1-sel.medianUs/all.medianUs,endToEndHealthCalculate:e2e};fs.writeFileSync(path.join(__dirname,'experiments','v6_final','runtime.json'),JSON.stringify(out,null,2)+'\n');console.log(JSON.stringify(out,null,2));
