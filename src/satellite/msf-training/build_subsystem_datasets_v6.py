#!/usr/bin/env python3
"""Build subsystem-local V6 datasets from the global coupled telemetry dataset."""
from __future__ import annotations
import argparse,re
from pathlib import Path
import pandas as pd

META=["scenarioId","split","tick","state","label","faultScope","faultKey","faultPair","faultMode","magnitude","startTick","readiness"]
DEFS={
  'power': {
    'candidateParams':['param_1','param_2','param_8','param_9','param_10'],
    'positiveParams':['param_1','param_2','param_8','param_9'],
    'pairs':['param_1|param_2','param_1|param_9','param_1|param_10','param_2|param_9','param_2|param_10','param_9|param_10','param_1|param_8'],
    'positivePairs':['param_1|param_2','param_1|param_9','param_1|param_10','param_2|param_9','param_2|param_10','param_9|param_10','param_1|param_8'],
  },
  'thermal': {
    'candidateParams':['param_3','param_4','param_9'],'positiveParams':['param_3','param_4','param_9'],
    'pairs':['param_3|param_4','param_3|param_9','param_4|param_9'],'positivePairs':['param_3|param_4','param_3|param_9','param_4|param_9'],
  },
  'attitude': {
    'candidateParams':['param_6','param_7','param_12'],'positiveParams':['param_6','param_7','param_12'],
    'pairs':['param_6|param_7','param_6|param_12','param_7|param_12'],'positivePairs':['param_6|param_7','param_6|param_12','param_7|param_12'],
  },
  'communication': {
    'candidateParams':['param_10','param_11'],'positiveParams':['param_11'],
    'pairs':['param_10|param_11'],'positivePairs':['param_10|param_11'],
  },
  'data': {
    'candidateParams':['param_1','param_2','param_9','param_10','param_11'],'positiveParams':['param_10'],
    'pairs':['param_1|param_10','param_2|param_10','param_9|param_10','param_10|param_11'],'positivePairs':[],
  },
  'propulsion': {
    'candidateParams':['param_5'],'positiveParams':['param_5'],'pairs':[],'positivePairs':[],
  },
}

def feature_columns(columns, spec):
    result=[]
    for c in columns:
        if c in META: continue
        if c.startswith('rel_'):
            m=re.match(r'rel_(param_\d+)_(param_\d+)__',c)
            if m and f'{m.group(1)}|{m.group(2)}' in spec['pairs']: result.append(c)
        else:
            m=re.match(r'(param_\d+)__',c)
            if m and m.group(1) in spec['candidateParams']: result.append(c)
    return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--out-dir',required=True)
    args=ap.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.data,low_memory=False)
    summary=[]
    for name,spec in DEFS.items():
        pos=(df.label==1)&(((df.faultScope=='univariate')&df.faultKey.isin(spec['positiveParams']))|((df.faultScope=='coupled')&df.faultPair.isin(spec['positivePairs'])))
        neg=(df.label==0); keep=pos|neg
        sub=df.loc[keep].copy(); sub['label']=pos.loc[keep].astype(int).to_numpy()
        feats=feature_columns(df.columns,spec); path=out/f'telemetry-msf-v6-{name}.csv'; sub[META+feats].to_csv(path,index=False)
        summary.append({'subsystem':name,'rows':len(sub),'D':len(feats),'positive':int(sub.label.sum()),'negative':int((sub.label==0).sum()),'file':path.name})
    sdf=pd.DataFrame(summary); sdf.to_csv(out/'subsystem-datasets-summary.csv',index=False); print(sdf.to_string(index=False))
if __name__=='__main__': main()
