#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('train_msf_mofs',HERE/'train_msf_mofs.py')
mod=importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name]=mod; SPEC.loader.exec_module(mod)
META=mod.META
SUBS=['power','thermal','attitude','data','communication']

def cmp(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float); d=a-b
    p=1.0 if np.allclose(d,0) else float(wilcoxon(a,b,alternative='two-sided').pvalue)
    return {'aMean':float(a.mean()),'bMean':float(b.mean()),'diff':float(d.mean()),'wins':int((d>1e-12).sum()),'ties':int((abs(d)<=1e-12).sum()),'losses':int((d<-1e-12).sum()),'p':p}

def main():
    out={'schemaVersion':'msf-v6-subsystem-comparison-1.0','subsystems':{}}
    rows=[]; fisher_rows=[]
    for sub in SUBS:
        df=pd.read_csv(HERE/f'telemetry-msf-v6-{sub}.csv',low_memory=False)
        feats=[c for c in df.columns if c not in META]
        tr=df[df.split=='train'].copy(); va=df[df.split=='val'].copy(); te=df[df.split=='test'].copy()
        med=tr[feats].median().fillna(0.0)
        for p in (tr,va,te): p.loc[:,feats]=p[feats].fillna(med)
        Xtr=tr[feats].to_numpy(float); ytr=tr.label.to_numpy(int); Xv=va[feats].to_numpy(float); yv=va.label.to_numpy(int); Xt=te[feats].to_numpy(float); yt=te.label.to_numpy(int)
        Xfit=np.vstack([Xtr,Xv]); yfit=np.r_[ytr,yv]
        fisher=mod.fisher_scores(Xtr,ytr); order=np.argsort(-fisher)
        exp=json.loads((HERE/'experiments'/'v6_final'/f'{sub}_92_101'/'experiment_results.json').read_text())
        sparse=json.loads((HERE/'experiments'/'v6_final'/f'{sub}_sparse_92_101'/'experiment_results.json').read_text())
        sp_by={r['seed']:r for r in sparse['runs']}
        msf_hv=[]; rnd_hv=[]; sp_hv=[]; msf_g=[]; fisher_g=[]; msf_k=[]; fisher_sets=[]
        for r in exp['runs']:
            seed=r['seed']; m=r['methods']['MSF']; rr=r['methods']['Random']; k=int(m['chosenFeatureCount'])
            fg=mod.holdout_gmean(Xfit,yfit,Xt,yt,order[:k])
            msf_hv.append(m['heldoutParetoHV']); rnd_hv.append(rr['heldoutParetoHV']); sp_hv.append(sp_by[seed]['heldoutParetoHV'])
            msf_g.append(m['chosenTestGmean']); fisher_g.append(fg); msf_k.append(k)
            fisher_rows.append({'subsystem':sub,'seed':seed,'K':k,'msfTestGmean':m['chosenTestGmean'],'fisherSameKTestGmean':fg,'difference':m['chosenTestGmean']-fg})
        all_g=mod.holdout_gmean(Xfit,yfit,Xt,yt,np.arange(len(feats)))
        info={
            'D':len(feats),'guidedCardinalities':mod.msf_cardinalities(len(feats)),'allFeaturesTestGmean':float(all_g),
            'msfVsRandomHV':cmp(msf_hv,rnd_hv),'msfVsSparseHV':cmp(msf_hv,sp_hv),'sparseVsRandomHV':cmp(sp_hv,rnd_hv),
            'msfVsFisherSameKTestGmean':cmp(msf_g,fisher_g),'meanMsfK':float(np.mean(msf_k)),
        }
        out['subsystems'][sub]=info
        rows.append({'subsystem':sub,'D':len(feats),'MSF_HV':np.mean(msf_hv),'Sparse_HV':np.mean(sp_hv),'Random_HV':np.mean(rnd_hv),'MSF_vs_Random_p':info['msfVsRandomHV']['p'],'MSF_vs_Sparse_p':info['msfVsSparseHV']['p'],'MSF_K':np.mean(msf_k),'MSF_G':np.mean(msf_g),'Fisher_sameK_G':np.mean(fisher_g),'MSF_vs_Fisher_p':info['msfVsFisherSameKTestGmean']['p'],'All_G':all_g})
    (HERE/'experiments'/'v6_final'/'subsystem_comparison.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    pd.DataFrame(rows).to_csv(HERE/'experiments'/'v6_final'/'subsystem_comparison.csv',index=False)
    pd.DataFrame(fisher_rows).to_csv(HERE/'experiments'/'v6_final'/'fisher_same_k.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__': main()
