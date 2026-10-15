"use strict";
/**
 * 卫星 V6 冻结特征核对与正确性自动测试（中文辅助脚本）。
 * 注意：使用原版模型 selectedFeatures，不会依据确认集重新选择特征。
 */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname,'../../src/satellite');
const {TelemetrySimulator} = require(path.join(root,'satellite-datastream-service/telemetry-simulator.js'));
const {extractFeatureVector,expectedFeatureNames} = require(path.join(root,'health-v2-service/core/health-features.js'));
const model = JSON.parse(fs.readFileSync(path.join(root,'health-v2-service/core/models/subsystem-health-model-v6.json'),'utf8'));
const map = JSON.parse(fs.readFileSync(path.resolve(__dirname,'../../configs/frozen_140_features.json'),'utf8'));
const dims={power:18,thermal:11,attitude:15,data:10,communication:69,propulsion:24};
const features=expectedFeatureNames(Array.from({length:12},(_,i)=>`param_${i+1}`));
const selected=[...new Set(Object.values(model.subsystems).flatMap(m=>m.selectedFeatures.map(f=>f.name)))];
assert.equal(features.length,582,'完整候选集应为582维');
assert.equal(selected.length,140,'历史冻结特征并集应为140维');
assert.equal(Object.values(dims).reduce((x,y)=>x+y,0),147);
for(const [k,n] of Object.entries(dims)) assert.equal(model.subsystems[k].selectedFeatures.length,n,`子系统 ${k} 数量不正确`);
assert.equal(new Set(features).size,582);
assert.deepEqual(selected,map.globalOrderedSelectedFeatureNames,'冻结映射与历史模型必须完全一致');
assert.deepEqual(map.globalIndices0Based,selected.map(n=>features.indexOf(n)),'582维索引对应关系不正确');
let assertions=0, comparisons=0,relation=0,maxError=0;
let latest;
function verify(tel,label){
  const all=extractFeatureVector(tel);
  const sparse=extractFeatureVector(tel,{selectedFeatureNames:selected,scales:[8,24,64]});
  for(const name of sparse.featureNames){
    assert(Object.hasOwn(all.features,name),`全量提取未返回 ${name}`);assertions++;
    const d=Math.abs(all.features[name]-sparse.features[name]);
    assert(d<=1e-12,`按需计算与全量计算不一致 ${label}/${name}/${d}`);assertions++;
    comparisons++;maxError=Math.max(maxError,d);
    if(name.startsWith('rel_'))relation++;
  }
  assert(sparse.featureNames.every(n=>selected.includes(n)));assertions++;
  return sparse;
}
const sim=new TelemetrySimulator({seed:424242,windowSize:64,scenarioId:'freeze-check'});
const checkpoints=new Set([1,7,8,23,24,63,64,80]);
for(let i=1;i<=80;i++){
  const pkt=sim.nextPoint(Date.UTC(2026,0,1));
  if(checkpoints.has(i)){latest=pkt.telemetry;verify(latest,`history_${i}`);}
}
const b=verify(latest,'causal-base');
const extra=JSON.parse(JSON.stringify(latest));
for(const x of extra.trendCharts||[]) if(Array.isArray(x.seriesData))x.seriesData.unshift(0,1,2,3);
const c=verify(extra,'extra-old-history');
for(const n of b.featureNames){assert.equal(b.features[n],c.features[n]);assertions++;}
const missing=JSON.parse(JSON.stringify(latest));
if(missing.trendCharts?.[0]?.seriesData?.length){missing.trendCharts[0].seriesData[15]=null;missing.trendCharts[1].seriesData[12]='missing';}
verify(missing,'null-and-missing');
const out={ok:true,historyLengths:[1,7,8,23,24,63,64,80],candidate:582,selected:140,subsystemCounts:dims,comparedFeatureValues:comparisons,crossChannelValues:relation,assertions,maxAbsError:maxError};
console.log(JSON.stringify(out,null,2));
