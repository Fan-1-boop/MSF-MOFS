#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, rankdata

PRIMARY=['power','thermal','attitude','data']
D={'power':267,'thermal':135,'attitude':135,'data':204}

def holm(ps):
    ps=np.asarray(ps,float); n=len(ps); order=np.argsort(ps); out=np.empty(n); running=0.0
    for i,idx in enumerate(order):
        running=max(running,(n-i)*ps[idx]); out[idx]=min(1.0,running)
    return out

def rbc(a,b):
    d=np.asarray(a,float)-np.asarray(b,float); d=d[np.abs(d)>1e-15]
    if len(d)==0:return 0.0
    ranks=rankdata(np.abs(d)); wp=ranks[d>0].sum(); wm=ranks[d<0].sum(); return float((wp-wm)/(wp+wm))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--results-dir',required=True); ap.add_argument('--out',required=True)
    args=ap.parse_args(); root=Path(args.results_dir); rows=[]
    for s in PRIMARY:
        a=pd.read_csv(root/s/'per_seed_results.csv').sort_values('seed')
        d=a.msf_hv-a.random_hv
        p=1.0 if np.allclose(d,0) else float(wilcoxon(a.msf_hv,a.random_hv,alternative='two-sided').pvalue)
        rows.append({'subsystem':s,'D':D[s],'MSF_HV_mean':a.msf_hv.mean(),'MSF_HV_std':a.msf_hv.std(ddof=1),
                     'Random_HV_mean':a.random_hv.mean(),'Random_HV_std':a.random_hv.std(ddof=1),'delta_HV':d.mean(),
                     'wins':int((d>0).sum()),'ties':int((d==0).sum()),'losses':int((d<0).sum()),'p_raw':p,'rbc':rbc(a.msf_hv,a.random_hv)})
    df=pd.DataFrame(rows); df['p_holm']=holm(df.p_raw.to_numpy()); Path(args.out).parent.mkdir(parents=True,exist_ok=True); df.to_csv(args.out,index=False)
    print(df.to_string(index=False))
if __name__=='__main__': main()
