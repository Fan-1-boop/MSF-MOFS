#!/usr/bin/env python3
"""Run matched MSF-vs-Random satellite feature-selection validation.

Input CSV must contain metadata columns plus candidate feature columns and a
scenario-level split column with train/val/test. Labels other than 0/1 are
excluded. No health-estimation code is used.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata
import msf_core as m


def nondominated_2d(points):
    uniq=sorted(set((float(x),float(y)) for x,y in points))
    out=[]
    for i,p in enumerate(uniq):
        if not any(j!=i and q[0]<=p[0] and q[1]<=p[1] and (q[0]<p[0] or q[1]<p[1]) for j,q in enumerate(uniq)):
            out.append(p)
    return sorted(out,key=lambda p:(p[0],-p[1]))


def hv2d_min(points, ref=(1.0,1.0)):
    front=nondominated_2d([(min(ref[0],max(0.0,x)),min(ref[1],max(0.0,y))) for x,y in points])
    hv=0.0; prev_y=ref[1]
    for x,y in front:
        if y<prev_y:
            hv+=(ref[0]-x)*(prev_y-y); prev_y=y
    return float(hv)


def heldout_hv(pop,Xfit,yfit,Xtest,ytest,D):
    points=[]
    for ind in m.rank_and_crowd(pop)[0]:
        cols=np.flatnonzero(ind.mask)
        g=0.0 if len(cols)==0 else m.holdout_gmean(Xfit,yfit,Xtest,ytest,cols)
        points.append((1.0-g,len(cols)/D))
    return hv2d_min(points,(1.0,1.0))


def rbc(a,b):
    d=np.asarray(a,float)-np.asarray(b,float); d=d[np.abs(d)>1e-15]
    if len(d)==0:return 0.0
    ranks=rankdata(np.abs(d)); wp=ranks[d>0].sum(); wm=ranks[d<0].sum()
    return float((wp-wm)/(wp+wm))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True)
    ap.add_argument('--out-dir',required=True)
    ap.add_argument('--seed-start',type=int,default=192)
    ap.add_argument('--seed-end',type=int,default=201)
    ap.add_argument('--search-samples',type=int,default=2400)
    ap.add_argument('--resume',action='store_true',help='Append only missing seeds; do not overwrite')
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.data,low_memory=False)
    feats=[c for c in df.columns if c not in m.META]
    df=df[df.label.isin([0,1])].copy(); df.label=df.label.astype(int)
    tr=df[df.split=='train'].copy(); va=df[df.split=='val'].copy(); te=df[df.split=='test'].copy()
    med=tr[feats].median().fillna(0.0)
    for p in (tr,va,te): p.loc[:,feats]=p[feats].fillna(med)
    Xtr=tr[feats].to_numpy(float); ytr=tr.label.to_numpy(int)
    Xv=va[feats].to_numpy(float); yv=va.label.to_numpy(int)
    Xt=te[feats].to_numpy(float); yt=te.label.to_numpy(int)
    Xfit=np.vstack([Xtr,Xv]); yfit=np.r_[ytr,yv]; D=len(feats)
    fisher=m.fisher_scores(Xtr,ytr); order=np.argsort(-fisher)
    per_seed_path=out/'per_seed_results.csv'
    rows=pd.read_csv(per_seed_path).to_dict('records') if args.resume and per_seed_path.exists() else []
    existing=[int(r['seed']) for r in rows]
    if len(existing)!=len(set(existing)):
        raise ValueError('Duplicate satellite seeds in existing per_seed_results.csv')
    done=set(existing)
    for seed in range(args.seed_start,args.seed_end+1):
        if seed in done: continue
        sx,sy=m.balanced_subsample(Xtr,ytr,args.search_samples,seed)
        rec={'seed':seed}
        for name,guided in [('MSF',True),('Random',False)]:
            pop,cache=m.evolve(sx,sy,order,seed=seed,guided=guided)
            chosen,_=m.choose_from_pareto(pop,Xtr,ytr,Xv,yv); idx=np.flatnonzero(chosen.mask)
            rec[name.lower()+'_hv']=heldout_hv(pop,Xfit,yfit,Xt,yt,D)
            rec[name.lower()+'_feature_count']=int(len(idx))
            rec[name.lower()+'_test_gmean']=float(m.holdout_gmean(Xfit,yfit,Xt,yt,idx))
            rec[name.lower()+'_unique_evals']=int(len(cache))
        rows.append(rec)
        pd.DataFrame(rows).sort_values('seed').to_csv(per_seed_path,index=False)
        print(f"seed={seed} MSF HV={rec['msf_hv']:.6f} Random HV={rec['random_hv']:.6f}")
    rdf=pd.DataFrame(rows).sort_values('seed')
    rdf.to_csv(per_seed_path,index=False)
    d=rdf.msf_hv-rdf.random_hv
    p=1.0 if np.allclose(d,0) else float(wilcoxon(rdf.msf_hv,rdf.random_hv,alternative='two-sided').pvalue)
    summary={'D':D,'guidedCardinalities':m.msf_cardinalities(D),'seeds':[args.seed_start,args.seed_end],
             'MSF_HV_mean':float(rdf.msf_hv.mean()),'MSF_HV_std':(float(rdf.msf_hv.std(ddof=1)) if len(rdf)>1 else None),
             'Random_HV_mean':float(rdf.random_hv.mean()),'Random_HV_std':(float(rdf.random_hv.std(ddof=1)) if len(rdf)>1 else None),
             'delta_HV':float(d.mean()),'wins':int((d>0).sum()),'ties':int((d==0).sum()),'losses':int((d<0).sum()),
             'wilcoxon_p_raw':p,'rank_biserial':rbc(rdf.msf_hv,rdf.random_hv)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
