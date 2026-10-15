from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata
from sklearn.model_selection import train_test_split

from irmofs.highdim_datasets import load_highdim_dataset
from irmofs.environments import make_nested_ir_environments
from irmofs.evaluator import FeatureSubsetEvaluator, heldout_gmeans
from irmofs.stable_guidance import fisher_scores, make_cisr_multiscale_population
from irmofs.highdim_formal import make_base_population
from irmofs.nsga2 import run_nsga2, hypervolume_2d
from irmofs.spea2 import run_spea2

CARRIERS = ["NSGA-II", "SPEA2"]
INITS = ["Random", "MSF"]


def front_test_metrics(pareto_pop, train_envs, X_test, y_test, k=5):
    objs=[]; means=[]; counts=[]
    for c in pareto_pop:
        tg=heldout_gmeans(c, train_envs, X_test, y_test, k)
        vals=np.asarray([tg[ir] for ir in sorted(tg)],float)
        mg=float(vals.mean()); means.append(mg); counts.append(int(c.sum()))
        objs.append([1.0-mg, float(np.mean(c))])
    F=np.asarray(objs,float)
    return dict(test_pf_hv=float(hypervolume_2d(F)), pf_size=int(len(pareto_pop)),
                best_test_mean_gmean_on_pf=float(np.max(means)), min_features_on_pf=int(np.min(counts)))


def rank_biserial(diff):
    d=np.asarray(diff,float); nz=d[~np.isclose(d,0,atol=1e-15,rtol=0)]
    if len(nz)==0: return 0.0
    r=rankdata(np.abs(nz),method='average'); wp=r[nz>0].sum(); wm=r[nz<0].sum()
    return float((wp-wm)/(wp+wm)) if wp+wm else 0.0


def write_stats(path,out):
    df=pd.read_csv(path)
    df.groupby(['dataset','carrier','init'],as_index=False).agg(
      n=('seed','count'), test_pf_hv_mean=('test_pf_hv','mean'), test_pf_hv_std=('test_pf_hv','std'),
      train_hv_mean=('train_hv','mean'), train_hv_std=('train_hv','std'),
      unique_evals_mean=('unique_wrapper_evaluations','mean'), total_time_mean=('total_time','mean')
    ).to_csv(out/'summary.csv',index=False)
    rows=[]
    for (ds,carrier),g in df.groupby(['dataset','carrier']):
        a=g[g['init']=='MSF'].set_index('seed')['test_pf_hv']; b=g[g['init']=='Random'].set_index('seed')['test_pf_hv']
        idx=a.index.intersection(b.index); diff=(a.loc[idx]-b.loc[idx]).to_numpy(float)
        if not len(diff): continue
        try: stat,p=wilcoxon(diff,zero_method='wilcox',alternative='two-sided',method='auto')
        except ValueError: stat,p=0.,1.
        rows.append(dict(dataset=ds,carrier=carrier,n=len(diff),mean_difference=diff.mean(),median_difference=np.median(diff),
                         wins=int((diff>1e-15).sum()),ties=int((np.abs(diff)<=1e-15).sum()),losses=int((diff<-1e-15).sum()),
                         rank_biserial=rank_biserial(diff),wilcoxon_stat=float(stat),p_raw=float(p)))
    pd.DataFrame(rows).to_csv(out/'paired_msf_vs_random.csv',index=False)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--datasets',nargs='+',default=['madelon','hill_valley','musk1','arcene'])
    ap.add_argument('--seeds',nargs='+',type=int,default=list(range(82,92))); ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True); ap.add_argument('--pop',type=int,default=12); ap.add_argument('--gen',type=int,default=6)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True); raw=a.out/'raw.csv'
    rows=pd.read_csv(raw).to_dict('records') if raw.exists() else []; done={(r['dataset'],int(r['seed']),r['carrier'],r['init']) for r in rows}
    proto={
      'purpose':'cross-MOEA plugin validation of the same MSF initializer',
      'development_seeds':a.seeds,'datasets':a.datasets,'population':a.pop,'generations':a.gen,
      'carrier_algorithms':['standard project NSGA-II','standard SPEA2 environmental archive selection'],
      'shared_variation':'uniform binary crossover p=0.9; independent bit-flip p=1/D',
      'matched_initialization':'same deterministic random base population; MSF replaces exactly first 3 chromosomes',
      'nominal_candidate_budget':f'N initial + G*N offspring = {a.pop*(a.gen+1)} candidates before cache duplicates',
      'wrapper':'same 5-NN / 3-fold / mean G-mean across IR={1,3,5}; feature-ratio second objective',
      'primary_test':'paired held-out Pareto-set HV, MSF vs Random within each carrier/dataset',
      'selection_rule':'no v2 method choice is made from this plugin experiment alone'
    }
    (a.out/'protocol.json').write_text(json.dumps(proto,indent=2),encoding='utf-8')
    for ds in a.datasets:
      bundle=load_highdim_dataset(ds,a.data_dir); D=bundle.X.shape[1]
      for seed in a.seeds:
        Xtr,Xte,ytr,yte=train_test_split(bundle.X,bundle.y,test_size=.30,stratify=bundle.y,random_state=seed)
        envs=make_nested_ir_environments(Xtr,ytr,[1,3,5],3,seed); scores=fisher_scores(Xtr,ytr)
        base=make_base_population(a.pop,D,seed,.20); msf,sizes=make_cisr_multiscale_population(base,scores,guided_fraction=3/a.pop)
        if len(sizes)!=3: raise RuntimeError(sizes)
        for carrier in CARRIERS:
          for init_name,init in [('Random',base),('MSF',msf)]:
            key=(ds,seed,carrier,init_name)
            if key in done: continue
            ev=FeatureSubsetEvaluator(envs,mode='mean',n_neighbors=5,backend='sklearn'); t0=time.perf_counter()
            if carrier=='NSGA-II': res=run_nsga2(ev.evaluate,D,a.pop,a.gen,random_state=seed,verbose=False,initial_population=init)
            else: res=run_spea2(ev.evaluate,D,a.pop,a.gen,archive_size=a.pop,random_state=seed,verbose=False,initial_population=init)
            search=time.perf_counter()-t0; train_hv=float(hypervolume_2d(res.pareto_objectives)); held0=time.perf_counter()
            fm=front_test_metrics(res.pareto_population,envs,Xte,yte,5); held=time.perf_counter()-held0
            rec=dict(dataset=ds,seed=seed,carrier=carrier,init=init_name,n_samples=len(bundle.y),n_features=D,
                     seed_sizes='/'.join(map(str,sizes)) if init_name=='MSF' else '',train_hv=train_hv,**fm,
                     unique_wrapper_evaluations=len(ev.cache),search_time=search,heldout_evaluation_time=held,total_time=search+held)
            rows.append(rec); done.add(key); pd.DataFrame(rows).to_csv(raw,index=False); write_stats(raw,a.out); print(json.dumps(rec),flush=True)
    write_stats(raw,a.out)
if __name__=='__main__': main()
