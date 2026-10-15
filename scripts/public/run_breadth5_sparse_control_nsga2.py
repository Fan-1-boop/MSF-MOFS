from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import wilcoxon, rankdata
from sklearn.model_selection import train_test_split

from irmofs.datasets import DatasetBundle, _clean_xy
from irmofs.environments import make_nested_ir_environments
from irmofs.evaluator import FeatureSubsetEvaluator, heldout_gmeans
from irmofs.stable_guidance import fisher_scores, make_cisr_multiscale_population
from irmofs.highdim_formal import make_base_population
from irmofs.nsga2 import run_nsga2, hypervolume_2d

DATASETS=['ALLAML','colon','GLI-85','Prostate-GE','SMK-CAN-187']
SEEDS=list(range(92,102))

def load_ds(name,data_dir):
    if name=='colon':
        df=pd.read_csv(data_dir/'pone.0246039.s002.csv')
        f=[c for c in df.columns if c.startswith('V')]
        X,y,names=_clean_xy(df[f],df['response']); return DatasetBundle(name,X,y,names)
    fmap={'ALLAML':'ALLAML.mat','GLI-85':'GLI-85.mat','Prostate-GE':'Prostate_GE.mat','SMK-CAN-187':'SMK_CAN_187.mat'}
    m=sio.loadmat(data_dir/fmap[name]); X,y,names=_clean_xy(m['X'],m['Y']); return DatasetBundle(name,X,y,names)

def front_metrics(pop,envs,Xte,yte):
    objs=[]; means=[]; worsts=[]; counts=[]
    for ch in pop:
        tg=heldout_gmeans(ch,envs,Xte,yte,5); v=np.array([tg[k] for k in sorted(tg)],float)
        mg=float(v.mean()); means.append(mg); worsts.append(float(v.min())); counts.append(int(ch.sum()))
        objs.append([1-mg,float(np.mean(ch))])
    return {'test_pf_hv':float(hypervolume_2d(np.asarray(objs,float))), 'pf_size':len(pop),
            'best_test_mean_gmean_on_pf':float(np.max(means)), 'best_test_worst_gmean_on_pf':float(np.max(worsts)),
            'min_features_on_pf':int(np.min(counts))}

def rb(diff):
    d=np.asarray(diff,float); nz=d[~np.isclose(d,0,atol=1e-15,rtol=0)]
    if len(nz)==0:return 0.0
    r=rankdata(np.abs(nz),method='average'); wp=r[nz>0].sum(); wm=r[nz<0].sum(); return float((wp-wm)/(wp+wm))

def holm(p):
    p=np.asarray(p,float); m=len(p); order=np.argsort(p); adj=np.empty(m); running=0.
    for j,i in enumerate(order): running=max(running,(m-j)*p[i]); adj[i]=min(1.,running)
    return adj

def stats(out):
    old=pd.read_csv('/mnt/data/v2pkg_tmp/MSF_MOFS_v2_9DATASET_FINAL_PACKAGE/untouched_breadth/raw.csv')
    old=old[old.carrier=='NSGA-II'].copy()
    new=pd.read_csv(out/'sparse_control_raw.csv')
    df=pd.concat([old,new],ignore_index=True,sort=False)
    df.to_csv(out/'random_sparse_msf_nsga2_raw.csv',index=False)
    sm=df.groupby(['dataset','init'],as_index=False).agg(n=('seed','count'),test_pf_hv_mean=('test_pf_hv','mean'),test_pf_hv_std=('test_pf_hv','std'),unique_evals_mean=('unique_wrapper_evaluations','mean'))
    sm.to_csv(out/'summary.csv',index=False)
    rows=[]
    for ds,g in df.groupby('dataset'):
        piv=g.pivot_table(index='seed',columns='init',values='test_pf_hv',aggfunc='first')
        for aa,bb,label in [('Sparse-control','Random','Sparse-control - Random'),('MSF','Sparse-control','MSF - Sparse-control'),('MSF','Random','MSF - Random')]:
            if aa not in piv.columns or bb not in piv.columns: continue
            d=(piv[aa]-piv[bb]).dropna().values
            try: st,p=wilcoxon(d,zero_method='wilcox',alternative='two-sided',method='auto')
            except ValueError: st,p=0.,1.
            rows.append({'dataset':ds,'contrast':label,'n':len(d),'mean_difference':float(d.mean()),'median_difference':float(np.median(d)),
                         'wins':int((d>1e-15).sum()),'ties':int((np.abs(d)<=1e-15).sum()),'losses':int((d<-1e-15).sum()),
                         'rank_biserial':rb(d),'wilcoxon_stat':float(st),'p_raw':float(p)})
    pr=pd.DataFrame(rows); pr['p_holm_across_5_within_contrast']=np.nan
    for c,idx in pr.groupby('contrast').groups.items(): pr.loc[idx,'p_holm_across_5_within_contrast']=holm(pr.loc[idx,'p_raw'].values)
    pr.to_csv(out/'paired_tests.csv',index=False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--data-dir',type=Path,default=Path('/mnt/data'))
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True); raw=a.out/'sparse_control_raw.csv'
    proto={'status':'LOCKED_BEFORE_BREADTH5_SPARSE_CONTROL_RESULTS','purpose':'isolate mixed-scale sparsity coverage from Fisher feature identity on the five untouched breadth datasets',
           'datasets':DATASETS,'seeds':SEEDS,'carrier':'NSGA-II','population':12,'generations':6,
           'sparse_control':'same deterministic Bernoulli(0.20) base population and same three frozen subset sizes as MSF; permute Fisher score vector across feature identities with RNG seed+9001 before top-k seeding; standard NSGA-II thereafter',
           'comparisons':['Sparse-control vs Random','MSF vs Sparse-control','MSF vs Random'],'primary_endpoint':'held-out Pareto-set HV','no_method_tuning':True}
    (a.out/'PROTOCOL_BEFORE_RESULTS.json').write_text(json.dumps(proto,indent=2),encoding='utf-8')
    rows=pd.read_csv(raw).to_dict('records') if raw.exists() else []; done={(r['dataset'],int(r['seed'])) for r in rows}
    for ds in DATASETS:
        b=load_ds(ds,a.data_dir); D=b.X.shape[1]
        for seed in SEEDS:
            if (ds,seed) in done: continue
            Xtr,Xte,ytr,yte=train_test_split(b.X,b.y,test_size=.30,stratify=b.y,random_state=seed)
            envs=make_nested_ir_environments(Xtr,ytr,[1,3,5],3,seed); scores=fisher_scores(Xtr,ytr)
            base=make_base_population(12,D,seed,.20)
            _,sizes=make_cisr_multiscale_population(base,scores,guided_fraction=3/12)
            ps=np.random.default_rng(seed+9001).permutation(scores)
            sparse,s2=make_cisr_multiscale_population(base,ps,guided_fraction=3/12)
            assert sizes==s2 and len(sizes)==3
            ev=FeatureSubsetEvaluator(envs,mode='mean',n_neighbors=5,backend='sklearn'); t0=time.perf_counter()
            res=run_nsga2(ev.evaluate,D,12,6,random_state=seed,verbose=False,initial_population=sparse)
            search=time.perf_counter()-t0; h0=time.perf_counter(); fm=front_metrics(res.pareto_population,envs,Xte,yte); held=time.perf_counter()-h0
            rec={'dataset':ds,'seed':seed,'carrier':'NSGA-II','init':'Sparse-control','n_samples':len(b.y),'n_features':D,'seed_sizes':'/'.join(map(str,sizes)),
                 'train_hv':float(hypervolume_2d(res.pareto_objectives)),**fm,'unique_wrapper_evaluations':len(ev.cache),'search_time':search,'heldout_evaluation_time':held,'total_time':search+held}
            rows.append(rec); done.add((ds,seed)); pd.DataFrame(rows).to_csv(raw,index=False); stats(a.out); print(json.dumps(rec),flush=True)
    stats(a.out)
if __name__=='__main__': main()
