#!/usr/bin/env python3
"""V6 Sparse-control preserving MSF cardinalities but randomizing feature identity."""
from __future__ import annotations
import argparse, sys, importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m; assert spec and spec.loader; spec.loader.exec_module(m); return m
m=load('train_msf_ctrl_v6', HERE/'train_msf_mofs.py')
r=load('multi_ctrl_v6', HERE/'run_multiseed_experiment.py')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default=str(HERE/'telemetry-msf-v6.csv'))
    ap.add_argument('--seed-start',type=int,default=92); ap.add_argument('--seed-end',type=int,default=101)
    ap.add_argument('--search-samples',type=int,default=2200)
    ap.add_argument('--out-dir',default=str(HERE/'experiments'/'v6_coupled'/'sparse_92_101'))
    args=ap.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.data,low_memory=False)
    features=[c for c in df.columns if c not in m.META]
    df=df[df.label.isin([0,1])].copy(); df.label=df.label.astype(int)
    tr=df[df.split=='train'].copy(); va=df[df.split=='val'].copy(); te=df[df.split=='test'].copy()
    med=tr[features].median().fillna(0)
    for part in (tr,va,te): part.loc[:,features]=part[features].fillna(med)
    Xtr=tr[features].to_numpy(float); ytr=tr.label.to_numpy(int)
    Xv=va[features].to_numpy(float); yv=va.label.to_numpy(int)
    Xt=te[features].to_numpy(float); yt=te.label.to_numpy(int)
    Xfit=np.vstack([Xtr,Xv]); yfit=np.r_[ytr,yv]; D=len(features)
    rows=[]; details=[]
    for seed in range(args.seed_start,args.seed_end+1):
        sx,sy=m.balanced_subsample(Xtr,ytr,args.search_samples,seed)
        sparse_order=np.random.default_rng(seed+50000).permutation(D)
        pop,cache=m.evolve(sx,sy,sparse_order,seed=seed,guided=True)
        chosen,_=m.choose_from_pareto(pop,Xtr,ytr,Xv,yv); idx=np.flatnonzero(chosen.mask)
        testg=m.holdout_gmean(Xfit,yfit,Xt,yt,idx)
        hv,records,nd=r.heldout_front_metrics(pop,Xfit,yfit,Xt,yt,D)
        rows.append({'seed':seed,'sparse_hv':hv,'sparse_feature_count':len(idx),'sparse_test_gmean':testg,'sparse_unique_evals':len(cache)})
        details.append({'seed':seed,'heldoutParetoHV':hv,'chosenFeatureCount':len(idx),'chosenTestGmean':testg,'chosenFeatures':[features[i] for i in idx]})
        print(f'seed={seed} Sparse HV={hv:.6f} G={testg:.4f} K={len(idx)}',flush=True)
    pd.DataFrame(rows).to_csv(out/'per_seed_results.csv',index=False)
    (out/'experiment_results.json').write_text(json.dumps({'runs':details},ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__': main()
