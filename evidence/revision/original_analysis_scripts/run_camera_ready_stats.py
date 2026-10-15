from pathlib import Path
import pandas as pd, numpy as np
from scipy.stats import wilcoxon, rankdata, spearmanr

ROOT=Path('/mnt/data/eic_work/writing_pkg/MSF_SATELLITE_PAPER_WRITING_PACKAGE/02_ORIGINAL_MSF_EVIDENCE/03_RESULTS')
OUT=Path('/mnt/data/eic_work/revision_analysis'); OUT.mkdir(exist_ok=True)
raw=pd.read_csv(ROOT/'combined_9dataset/raw_9dataset.csv')
refs=[(1.0,1.0),(1.05,1.05),(1.10,1.10),(1.10,1.05),(1.05,1.10)]

def holm(pvals):
    p=np.asarray(pvals,float); m=len(p); order=np.argsort(p); adj=np.empty(m)
    running=0.0
    for j,idx in enumerate(order):
        val=(m-j)*p[idx]
        running=max(running,val)
        adj[idx]=min(1.0,running)
    return adj

def rb_paired(d):
    d=np.asarray(d,float); d=d[np.isfinite(d)&(d!=0)]
    if len(d)==0:return 0.0
    r=rankdata(np.abs(d),method='average')
    pos=r[d>0].sum(); neg=r[d<0].sum(); denom=pos+neg
    return (pos-neg)/denom if denom else 0.0

def wtl(d,tol=1e-12):
    d=np.asarray(d,float)
    return int((d>tol).sum()),int((np.abs(d)<=tol).sum()),int((d<-tol).sum())

# HV sensitivity
sens_rows=[]
for r1,r2 in refs:
    a=r1-1; b=r2-1
    x=raw.copy()
    minf1=1-x['best_test_mean_gmean_on_pf']
    minf2=x['min_features_on_pf']/x['n_features']
    x['hv_ref']=x['test_pf_hv'] + a*(1-minf2) + b*(1-minf1) + a*b
    for (dataset,carrier),g in x.groupby(['dataset','carrier']):
        piv=g.pivot(index='seed',columns='init',values='hv_ref').dropna()
        d=(piv['MSF']-piv['Random']).values
        try:p=wilcoxon(d,alternative='two-sided',zero_method='wilcox').pvalue
        except ValueError:p=1.0
        W,T,L=wtl(d)
        sens_rows.append(dict(ref=f'({r1:.2f},{r2:.2f})',dataset=dataset,carrier=carrier,n=len(d),random_mean=piv['Random'].mean(),msf_mean=piv['MSF'].mean(),mean_delta=d.mean(),median_delta=np.median(d),wins=W,ties=T,losses=L,p_raw=p,rbc=rb_paired(d)))
sens=pd.DataFrame(sens_rows)
sens['p_holm_within_ref_carrier']=np.nan
for (ref,carrier),idx in sens.groupby(['ref','carrier']).groups.items():
    sens.loc[idx,'p_holm_within_ref_carrier']=holm(sens.loc[idx,'p_raw'])
sens.to_csv(OUT/'hv_reference_sensitivity.csv',index=False)
# summary invariance vs baseline
summ=[]
for (ref,carrier),g in sens.groupby(['ref','carrier']):
    summ.append(dict(ref=ref,carrier=carrier,datasets=len(g),positive_mean_delta=int((g.mean_delta>0).sum()),negative_mean_delta=int((g.mean_delta<0).sum()),holm_sig_positive=int(((g.mean_delta>0)&(g.p_holm_within_ref_carrier<.05)).sum()),holm_sig_negative=int(((g.mean_delta<0)&(g.p_holm_within_ref_carrier<.05)).sum()),total_seed_wins=int(g.wins.sum()),total_seed_ties=int(g.ties.sum()),total_seed_losses=int(g.losses.sum()),mean_dataset_delta=g.mean_delta.mean()))
pd.DataFrame(summ).to_csv(OUT/'hv_reference_sensitivity_summary.csv',index=False)

# paired CI/effect table at original ref
cirows=[]; rng=np.random.default_rng(4302026)
for (dataset,carrier),g in raw.groupby(['dataset','carrier']):
    piv=g.pivot(index='seed',columns='init',values='test_pf_hv').dropna(); d=(piv['MSF']-piv['Random']).values
    boot=np.array([rng.choice(d,size=len(d),replace=True).mean() for _ in range(20000)])
    lo,hi=np.quantile(boot,[.025,.975])
    try:p=wilcoxon(d,alternative='two-sided',zero_method='wilcox').pvalue
    except ValueError:p=1.0
    W,T,L=wtl(d)
    cirows.append(dict(dataset=dataset,carrier=carrier,n=len(d),random_mean=piv.Random.mean(),msf_mean=piv.MSF.mean(),mean_delta=d.mean(),ci95_low=lo,ci95_high=hi,wins=W,ties=T,losses=L,p_raw=p,rbc=rb_paired(d)))
ci=pd.DataFrame(cirows)
ci['p_holm_within_carrier']=np.nan
for carrier,idx in ci.groupby('carrier').groups.items():ci.loc[idx,'p_holm_within_carrier']=holm(ci.loc[idx,'p_raw'])
ci.to_csv(OUT/'paired_hv_ci_effect_9dataset.csv',index=False)

# sparse identity gains (formal n=5), preserved for reliability join if ranking data becomes available
sp=pd.read_csv(ROOT/'sparse_control_attribution/random_sparse_msf_nsga2_raw.csv')
p=sp.pivot_table(index=['dataset','seed'],columns='init',values='test_pf_hv').reset_index()
if {'MSF','Sparse-control'}.issubset(p.columns):
    p['identity_gain']=p['MSF']-p['Sparse-control']
    agg=p.groupby('dataset').identity_gain.agg(['mean','median','std','count']).reset_index()
    agg.to_csv(OUT/'sparse_identity_gain_by_dataset.csv',index=False)

print('WROTE',OUT)
print('\nHV sensitivity summary:')
print(pd.read_csv(OUT/'hv_reference_sensitivity_summary.csv').to_string(index=False))
print('\nCI/effect:')
print(ci[['dataset','carrier','mean_delta','ci95_low','ci95_high','wins','ties','losses','p_holm_within_carrier','rbc']].to_string(index=False))
print('\nSparse identity gains:')
print(pd.read_csv(OUT/'sparse_identity_gain_by_dataset.csv').to_string(index=False))
