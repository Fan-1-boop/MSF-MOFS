"""Deterministic post-hoc held-out PF-HV reconstruction for archived P0-3 runs.
This does not alter or re-tune the frozen evolutionary search. Source data and code
come solely from the user's supplied MSF-MOFS source/evidence archive.
"""
import sys, csv, json, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
ROOT=Path('/mnt/data/heldout_audit_work/MSF_MOFS_EIC2026_PAPER_AND_SATELLITE_VALIDATION_PACKAGE')
sys.path.insert(0,str(ROOT/'03_PUBLIC_MSF_SOURCE'))
from runners.run_p0_3_mechanism import run_nsga2_snapshots, front_test_hv
from irmofs.highdim_datasets import load_highdim_dataset
from irmofs.environments import make_nested_ir_environments
from irmofs.evaluator import FeatureSubsetEvaluator
from irmofs.stable_guidance import fisher_scores, make_cisr_multiscale_population
from irmofs.highdim_formal import make_base_population
from irmofs.nsga2 import hypervolume_2d
DATA_DIR=ROOT/'04_PUBLIC_MSF_DATA/core_original_archives'
ARCHIVE_RAW=ROOT/'05_PUBLIC_MSF_EVIDENCE/P0_72_81_AND_MECHANISM/P0_3/population_diversity_raw.csv'
OUT=Path('/mnt/data/heldout_audit_work/reconstructed_heldout_runs.csv')

def run_one(ds,seed,method):
    archive=pd.read_csv(ARCHIVE_RAW)
    obs=archive[(archive.dataset==ds)&(archive.seed==seed)&(archive.method==method)].iloc[0]
    bundle=load_highdim_dataset(ds,DATA_DIR); D=bundle.X.shape[1]
    Xtr,Xte,ytr,yte=train_test_split(bundle.X,bundle.y,test_size=.30,stratify=bundle.y,random_state=seed)
    envs=make_nested_ir_environments(Xtr,ytr,[1,3,5],3,seed)
    scores=fisher_scores(Xtr,ytr)
    base=make_base_population(12,D,seed,.20)
    msf,sizes=make_cisr_multiscale_population(base,scores,guided_fraction=3/12)
    perm_scores=np.random.default_rng(seed+9001).permutation(scores)
    sparse,sizes2=make_cisr_multiscale_population(base,perm_scores,guided_fraction=3/12)
    assert sizes==sizes2 and len(sizes)==3
    pop={'Random':base,'Sparse-control':sparse,'MSF-MOFS':msf}[method]
    ev=FeatureSubsetEvaluator(envs,mode='mean',n_neighbors=5,backend='sklearn')
    t=time.perf_counter()
    res,_=run_nsga2_snapshots(ev.evaluate,D,12,6,seed,pop)
    train_hv=float(hypervolume_2d(res.pareto_objectives))
    original=float(obs.train_hv_final)
    match=abs(train_hv-original)
    if match>1e-9:
        raise AssertionError(f'FAILED archived train-HV reproduction: {ds} {seed} {method} archived={original} rerun={train_hv} diff={match}')
    thv=front_test_hv(res.pareto_population,envs,Xte,yte,5)
    recorded=obs.test_pf_hv_final
    if pd.notna(recorded) and abs(float(recorded)-thv)>1e-9:
        raise AssertionError(f'FAILED held-out cross-check: {ds} {seed} {method}, original={recorded}, got={thv}')
    return dict(dataset=ds,seed=seed,method=method,train_hv_replayed=train_hv,train_hv_archived=original,train_hv_abs_error=match,test_pf_hv_replayed=thv,test_pf_hv_archived=('' if pd.isna(recorded) else float(recorded)),pf_size=len(res.pareto_population),unique_wrapper_evals=len(ev.cache),guided_sizes='/'.join(map(str,sizes)) if method!='Random' else '',elapsed_seconds=round(time.perf_counter()-t,3))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--datasets',nargs='*',default=['madelon','arcene']);ap.add_argument('--methods',nargs='*',default=['MSF-MOFS','Sparse-control']);ap.add_argument('--seeds',nargs='*',type=int,default=list(range(72,82)));a=ap.parse_args()
    fieldnames=['dataset','seed','method','train_hv_replayed','train_hv_archived','train_hv_abs_error','test_pf_hv_replayed','test_pf_hv_archived','pf_size','unique_wrapper_evals','guided_sizes','elapsed_seconds']
    done=set()
    if OUT.exists():
        with OUT.open() as f:
            done={(r['dataset'],int(r['seed']),r['method']) for r in csv.DictReader(f)}
    for ds in a.datasets:
        for seed in a.seeds:
            for method in a.methods:
                if (ds,seed,method) in done: continue
                row=run_one(ds,seed,method)
                with OUT.open('a',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=fieldnames)
                    if f.tell()==0:w.writeheader()
                    w.writerow(row)
                print(json.dumps(row),flush=True)
