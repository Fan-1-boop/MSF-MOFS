from __future__ import annotations
import csv, json, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import wilcoxon, rankdata
from sklearn.model_selection import train_test_split

from irmofs.datasets import DatasetBundle, _clean_xy
from irmofs.highdim_datasets import load_highdim_dataset
from irmofs.environments import make_nested_ir_environments
from irmofs.evaluator import FeatureSubsetEvaluator, heldout_gmeans
from irmofs.stable_guidance import fisher_scores, make_cisr_multiscale_population
from irmofs.highdim_formal import make_base_population
from irmofs.nsga2 import fast_non_dominated_sort, hypervolume_2d
from run_p0_3_mechanism import run_nsga2_snapshots

DATASETS=['hill_valley','Prostate-GE']
SEEDS=list(range(72,82))
METHODS=['Random','MSF-MOFS']
POP=50
GEN=1


def load_dataset(name: str, core_dir: Path, untouched_dir: Path) -> DatasetBundle:
    if name == 'hill_valley':
        b = load_highdim_dataset('hill_valley', core_dir)
        if b.X.shape != (1212,100):
            raise RuntimeError(f'hill_valley shape {b.X.shape} != (1212,100)')
        return b
    if name == 'Prostate-GE':
        p = untouched_dir / 'Prostate_GE.mat'
        m = sio.loadmat(p)
        if 'X' not in m or 'Y' not in m:
            raise ValueError(f'{p} missing X/Y')
        X,y,names = _clean_xy(m['X'],m['Y'])
        b = DatasetBundle('Prostate-GE',X,y,names)
        if b.X.shape != (102,5966):
            raise RuntimeError(f'Prostate-GE shape {b.X.shape} != (102,5966)')
        return b
    raise ValueError(name)


def front_metrics(pop,F,train_envs,Xtest,ytest):
    _,ranks=fast_non_dominated_sort(F)
    pf=pop[ranks==0]
    objs=[]
    for ch in pf:
        tg=heldout_gmeans(ch,train_envs,Xtest,ytest,5)
        vals=np.array([tg[k] for k in sorted(tg)],float)
        objs.append([1-float(vals.mean()),float(ch.mean())])
    return float(hypervolume_2d(np.asarray(objs,float))),int(len(pf))


def write_csv(path,rows):
    if not rows:return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def rbc(diff):
    d=np.asarray(diff,float); nz=d[~np.isclose(d,0,atol=1e-15,rtol=0)]
    if not len(nz):return 0.0
    rr=rankdata(np.abs(nz),method='average');wp=rr[nz>0].sum();wm=rr[nz<0].sum()
    return float((wp-wm)/(wp+wm))


def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--core-dir',type=Path,required=True)
    ap.add_argument('--untouched-dir',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--datasets',nargs='+',choices=DATASETS,default=DATASETS)
    ap.add_argument('--seeds',nargs='+',type=int,default=SEEDS)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    rawp=a.out/'large_population_extension_raw.csv'
    rows=[]
    if rawp.exists():
        with rawp.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    done={(r['dataset'],int(r['seed']),r['method']) for r in rows}
    for ds in a.datasets:
        b=load_dataset(ds,a.core_dir,a.untouched_dir);D=b.X.shape[1]
        for seed in a.seeds:
            Xtr,Xte,ytr,yte=train_test_split(b.X,b.y,test_size=.30,stratify=b.y,random_state=seed)
            env=make_nested_ir_environments(Xtr,ytr,[1,3,5],3,seed)
            scores=fisher_scores(Xtr,ytr)
            base=make_base_population(POP,D,seed,.20)
            msf,sizes=make_cisr_multiscale_population(base,scores,guided_fraction=3.0/POP)
            if len(sizes)!=3: raise RuntimeError(sizes)
            for method,init in [('Random',base),('MSF-MOFS',msf)]:
                if (ds,seed,method) in done:continue
                ev=FeatureSubsetEvaluator(env,mode='mean',n_neighbors=5,backend='sklearn')
                t0=time.perf_counter()
                res,snaps=run_nsga2_snapshots(ev.evaluate,D,POP,GEN,seed,init)
                search=time.perf_counter()-t0
                F0=np.vstack([ev.evaluate(x) for x in snaps[0]])
                g0_unique=int(len(np.unique(snaps[0],axis=0)))
                hv0,pf0=front_metrics(snaps[0],F0,env,Xte,yte)
                hv1,pf1=front_metrics(res.population,res.objectives,env,Xte,yte)
                rec={
                  'dataset':ds,'seed':seed,'method':method,'population':POP,'guided_chromosomes':0 if method=='Random' else 3,
                  'guided_fraction':0.0 if method=='Random' else 3.0/POP,'guided_sizes':'' if method=='Random' else '/'.join(map(str,sizes)),
                  'gen0_unique_wrapper_evaluations':g0_unique,'gen0_test_pf_hv':hv0,'gen0_pf_size':pf0,
                  'gen1_unique_wrapper_evaluations':int(len(ev.cache)),'gen1_test_pf_hv':hv1,'gen1_pf_size':pf1,
                  'gen1_train_hv':float(hypervolume_2d(res.pareto_objectives)),'search_wall_seconds_for_gen1_run':search,
                }
                rows.append(rec);done.add((ds,seed,method));write_csv(rawp,rows)
                print(json.dumps(rec,ensure_ascii=False),flush=True)
    summary=[];tests=[]
    for ds in a.datasets:
      for method in METHODS:
       rr=[r for r in rows if r['dataset']==ds and r['method']==method]
       for stage in ['gen0','gen1']:
        hv=np.array([float(r[f'{stage}_test_pf_hv']) for r in rr]);ev=np.array([float(r[f'{stage}_unique_wrapper_evaluations']) for r in rr])
        summary.append({'dataset':ds,'method':method,'stage':stage,'n':len(hv),'test_pf_hv_mean':float(hv.mean()),'test_pf_hv_std':float(hv.std(ddof=1)),'unique_wrapper_evaluations_mean':float(ev.mean()),'unique_wrapper_evaluations_std':float(ev.std(ddof=1))})
      A={int(r['seed']):r for r in rows if r['dataset']==ds and r['method']=='MSF-MOFS'}
      B={int(r['seed']):r for r in rows if r['dataset']==ds and r['method']=='Random'}
      for stage in ['gen0','gen1']:
        d=np.array([float(A[s][f'{stage}_test_pf_hv'])-float(B[s][f'{stage}_test_pf_hv']) for s in a.seeds])
        try:stat,p=wilcoxon(d,zero_method='wilcox',alternative='two-sided',method='auto');stat,p=float(stat),float(p)
        except ValueError:stat,p=0.,1.
        tests.append({'dataset':ds,'candidate':'MSF-MOFS','baseline':'Random','stage':stage,'n':len(d),'mean_difference':float(d.mean()),'median_difference':float(np.median(d)),'wins':int((d>1e-15).sum()),'ties':int((np.abs(d)<=1e-15).sum()),'losses':int((d<-1e-15).sum()),'rank_biserial':rbc(d),'wilcoxon_stat':stat,'p_raw':p})
    write_csv(a.out/'large_population_extension_summary.csv',summary);write_csv(a.out/'large_population_extension_paired_tests.csv',tests)

if __name__=='__main__':main()
