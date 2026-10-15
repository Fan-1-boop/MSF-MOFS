"""Replay 40 seed-level MSF/Sparse searches for reliability vs held-out gain.

Uses original standard NSGA-II evaluation protocol. The archived results are
comparison-only, never loaded as search inputs. Restart-safe append per cell.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from irmofs.highdim_datasets import load_highdim_dataset
from irmofs.environments import make_nested_ir_environments
from irmofs.evaluator import FeatureSubsetEvaluator
from irmofs.stable_guidance import fisher_scores,make_cisr_multiscale_population
from irmofs.highdim_formal import make_base_population
from irmofs.nsga2 import hypervolume_2d
from run_p0_3_mechanism import run_nsga2_snapshots,front_test_hv

DATASETS=('madelon','arcene');SEEDS=tuple(range(72,82));METHODS=('MSF-MOFS','Sparse-control')
COLUMNS=['dataset','seed','method','train_hv_replayed','train_hv_archived','train_hv_abs_error','test_pf_hv_replayed','test_pf_hv_archived','pf_size','unique_wrapper_evals','guided_sizes','elapsed_seconds']


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--out',type=Path,default=None)
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS))
    p.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
    a=p.parse_args(argv);root=a.root.resolve()
    dest=a.out or root/'results/reference/reliability';dest.mkdir(parents=True,exist_ok=True)
    raw=dest/'reconstructed_heldout_runs.csv'
    archived=root/'evidence/public/P0_72_81_AND_MECHANISM/P0_3/population_diversity_raw.csv'
    baseline=pd.read_csv(archived)
    original=root/'evidence/revision/v6_heldout_reliability/reconstructed_heldout_runs.csv'
    previous=pd.read_csv(original).set_index(['dataset','seed','method'],verify_integrity=True)
    rows=list(csv.DictReader(raw.open(newline='',encoding='utf-8'))) if raw.exists() else []
    seen={(r['dataset'],int(r['seed']),r['method']) for r in rows}
    if len(rows)!=len(seen):raise ValueError('Duplicated seed keys in replay output')
    for ds in a.datasets:
        b=load_highdim_dataset(ds,root/'data/public/core');D=b.X.shape[1]
        for seed in a.seeds:
            Xtr,Xte,ytr,yte=train_test_split(b.X,b.y,test_size=.30,stratify=b.y,random_state=seed)
            env=make_nested_ir_environments(Xtr,ytr,[1,3,5],3,seed)
            base=make_base_population(12,D,seed,.20)
            scores=fisher_scores(Xtr,ytr)
            msf,sizes=make_cisr_multiscale_population(base,scores,guided_fraction=3/12)
            permuted=np.random.default_rng(seed+9001).permutation(scores)
            sparse,sizes2=make_cisr_multiscale_population(base,permuted,guided_fraction=3/12)
            assert sizes==sizes2 and len(sizes)==3
            for method,pop in (('MSF-MOFS',msf),('Sparse-control',sparse)):
                key=(ds,seed,method)
                if key in seen:continue
                t=time.perf_counter()
                ev=FeatureSubsetEvaluator(env,mode='mean',n_neighbors=5,backend='sklearn')
                res,_=run_nsga2_snapshots(ev.evaluate,D,12,6,seed,pop)
                train_hv=float(hypervolume_2d(res.pareto_objectives))
                holdout=float(front_test_hv(res.pareto_population,env,Xte,yte,5))
                ref=baseline[(baseline.dataset==ds)&(baseline.seed==seed)&(baseline.method==method)]
                if len(ref)!=4:raise RuntimeError(f'Expected 4 archived snapshot rows for {key}, got {len(ref)}')
                orig=float(ref.iloc[0].train_hv_final)
                abs_error=abs(train_hv-orig)
                if abs_error>1e-9:raise AssertionError(f'Train HV mismatch {key}: {train_hv} vs {orig}')
                old=previous.loc[key]
                if abs(holdout-float(old.test_pf_hv_replayed))>1e-9:
                    raise AssertionError(f'Held-out HV mismatch {key}: {holdout} vs {old.test_pf_hv_replayed}')
                hist=ref.iloc[0].test_pf_hv_final
                if pd.notna(hist) and abs(holdout-float(hist))>1e-9:
                    raise AssertionError(f'Original held-out mismatch for {key}')
                rec={'dataset':ds,'seed':seed,'method':method,'train_hv_replayed':train_hv,
                    'train_hv_archived':orig,'train_hv_abs_error':abs_error,
                    'test_pf_hv_replayed':holdout,'test_pf_hv_archived':('' if pd.isna(hist) else float(hist)),
                    'pf_size':len(res.pareto_population),'unique_wrapper_evals':len(ev.cache),
                    'guided_sizes':'/'.join(map(str,sizes)),
                    'elapsed_seconds':round(time.perf_counter()-t,3)}
                with raw.open('a',newline='',encoding='utf-8') as f:
                    writer=csv.DictWriter(f,fieldnames=COLUMNS)
                    if f.tell()==0:writer.writeheader()
                    writer.writerow(rec)
                rows.append(rec);seen.add(key)
                print(json.dumps(rec,ensure_ascii=False),flush=True)
    wanted={(ds,seed,method) for ds in DATASETS for seed in SEEDS for method in METHODS}
    completed=seen&wanted
    report={'new_search_rows':len(completed),'expected_rows':len(wanted),
        'complete':len(completed)==len(wanted),'matches_historical_heldout_and_train':True,
        'historical_replay_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
        'original_p0_3_sha256':hashlib.sha256(archived.read_bytes()).hexdigest()}
    (dest/'REPLAY_STATUS.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('REPLAY:',json.dumps(report),flush=True)

if __name__=='__main__':main()
