"""Portable frozen Sparse-control run for the 5 untouched public datasets.

Scientific search behavior: copied from run_breadth5_sparse_control_nsga2.py;
only path handling, resume and postprocessing are changed.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from irmofs.environments import make_nested_ir_environments
from irmofs.evaluator import FeatureSubsetEvaluator
from irmofs.stable_guidance import fisher_scores, make_cisr_multiscale_population
from irmofs.highdim_formal import make_base_population
from irmofs.nsga2 import run_nsga2, hypervolume_2d
from run_breadth5_sparse_control_nsga2 import load_ds, front_metrics, DATASETS, SEEDS


def main():
    ap=argparse.ArgumentParser(description='5-dataset Sparse-control: original search, portable paths')
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    args=ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    target=args.out/'sparse_control_raw.csv'
    rows=pd.read_csv(target).to_dict('records') if target.exists() else []
    keys=[(r['dataset'],int(r['seed'])) for r in rows]
    if len(keys)!=len(set(keys)):
        raise ValueError('Duplicate Sparse-control dataset/seed in result file')
    done=set(keys)
    protocol={
       'datasets':args.datasets, 'seeds':args.seeds, 'carrier':'NSGA-II',
       'population':12,'generations':6,
       'base_population':'Bernoulli(0.20), seed=seed',
       'guidance':'three chromosomes at fixed MSF sizes',
       'identity':'np.random.default_rng(seed+9001).permutation(fisher_scores(X_train,y_train))',
       'outer_split':'train_test_split(test_size=.30,stratify=y,random_state=seed)',
       'evaluation':'held-out Pareto-set HV; training-only CV scaling and 5NN',
       'source':'scripts/public/run_breadth5_sparse_control_nsga2.py (search logic unchanged)',
    }
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    for ds in args.datasets:
        b=load_ds(ds,args.data_dir)
        D=b.X.shape[1]
        for seed in args.seeds:
            if (ds,seed) in done: continue
            Xtr,Xte,ytr,yte=train_test_split(b.X,b.y,test_size=.30,stratify=b.y,random_state=seed)
            envs=make_nested_ir_environments(Xtr,ytr,[1,3,5],3,seed)
            scores=fisher_scores(Xtr,ytr)
            base=make_base_population(12,D,seed,.20)
            _,sizes=make_cisr_multiscale_population(base,scores,guided_fraction=3/12)
            ps=np.random.default_rng(seed+9001).permutation(scores)
            sparse,s2=make_cisr_multiscale_population(base,ps,guided_fraction=3/12)
            assert sizes==s2 and len(sizes)==3
            ev=FeatureSubsetEvaluator(envs,mode='mean',n_neighbors=5,backend='sklearn')
            t0=time.perf_counter()
            res=run_nsga2(ev.evaluate,D,12,6,random_state=seed,verbose=False,initial_population=sparse)
            search=time.perf_counter()-t0
            h0=time.perf_counter()
            fm=front_metrics(res.pareto_population,envs,Xte,yte)
            held=time.perf_counter()-h0
            rec={'dataset':ds,'seed':seed,'carrier':'NSGA-II','init':'Sparse-control',
                 'n_samples':len(b.y),'n_features':D,'seed_sizes':'/'.join(map(str,sizes)),
                 'train_hv':float(hypervolume_2d(res.pareto_objectives)),**fm,
                 'unique_wrapper_evaluations':len(ev.cache),
                 'search_time':search,'heldout_evaluation_time':held,'total_time':search+held}
            rows.append(rec); done.add((ds,seed))
            # checkpoint after every search to support safe manual resumption
            pd.DataFrame(rows).to_csv(target,index=False)
            print(json.dumps(rec,ensure_ascii=False),flush=True)
    print(f'Sparse-control recorded {len(rows)} unique experiments in {target}')
    # Strict stats and archive comparison live in msf_repro.validation_v3;
    # do not silently generate incomplete 5-dataset statistics here.

if __name__=='__main__':main()
