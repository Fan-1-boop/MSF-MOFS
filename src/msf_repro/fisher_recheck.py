"""Independent recomputation of archived Fisher rank-stability records.

Reads original public datasets, recreates training splits and recomputes Fisher
rankings. The archive is used *only for comparison*, never as source of values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from irmofs.highdim_datasets import load_highdim_dataset
from irmofs.stable_guidance import fisher_scores

DATASETS = ('madelon','arcene')
SEEDS = tuple(range(72,82))


def build_pairwise(data_dir:Path) -> pd.DataFrame:
    records=[]
    for ds in DATASETS:
        bundle=load_highdim_dataset(ds,data_dir)
        d=bundle.X.shape[1]
        ks=[]
        for val in (10, 0.05*d, 0.10*d, 0.20*d):
            k=max(1,min(d,int(round(val))))
            if k not in ks: ks.append(k)
        ranks={};tops={}
        for seed in SEEDS:
            Xtr,_,ytr,_=train_test_split(bundle.X,bundle.y,test_size=.30,stratify=bundle.y,random_state=seed)
            order=np.argsort(-fisher_scores(Xtr,ytr),kind='stable')
            ranks[seed]=np.empty(d,dtype=np.int64)
            ranks[seed][order]=np.arange(1,d+1)
            tops[seed]={k:set(map(int,order[:k])) for k in ks}
        for a,b in combinations(SEEDS,2):
            records.append(dict(dataset=ds,seed_a=a,seed_b=b,metric='spearman_rank',k=np.nan,value=float(spearmanr(ranks[a],ranks[b]).statistic),k_fraction=np.nan))
            for k in ks:
                A=tops[a][k];B=tops[b][k]
                records.append(dict(dataset=ds,seed_a=a,seed_b=b,metric='topk_jaccard',k=float(k),value=len(A&B)/len(A|B),k_fraction=k/d))
    return pd.DataFrame(records)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--out',type=Path,default=None)
    a=p.parse_args(argv);root=a.root.resolve()
    out=a.out or root/'results/reproduced/paper/fisher_stability';out.mkdir(parents=True,exist_ok=True)
    archived=root/'evidence/public/P0_72_81_AND_MECHANISM/P0_3/fisher_stability_pairwise.csv'
    generated=build_pairwise(root/'data/public/core')
    expected=pd.read_csv(archived)
    keys=['dataset','seed_a','seed_b','metric','k']
    if len(generated)!=len(expected):raise RuntimeError(f'Fisher row mismatch: {len(generated)} vs {len(expected)}')
    new=generated.sort_values(keys,na_position='first').reset_index(drop=True)
    old=expected.sort_values(keys,na_position='first').reset_index(drop=True)
    same_keys=new[keys].fillna(-1).equals(old[keys].fillna(-1))
    vals=np.abs(new['value'].to_numpy(float)-old['value'].to_numpy(float))
    scales=np.abs(new['k_fraction'].fillna(-1).to_numpy(float)-old['k_fraction'].fillna(-1).to_numpy(float))
    max_error=float(np.max(vals))
    ok=same_keys and max_error<=1e-12 and float(np.max(scales))<=1e-12
    generated.to_csv(out/'fisher_stability_pairwise_recomputed.csv',index=False,float_format='%.17g')
    result={'status':'MATCH' if ok else 'MISMATCH','evidence_type':'independent_from_raw_datasets',
        'n_rows':len(new),'n_datasets':len(DATASETS),'n_outer_seeds_per_dataset':len(SEEDS),
        'max_abs_jaccard_or_spearman_error':max_error,'keys_equal':bool(same_keys),
        'source_hash_sha256':hashlib.sha256(archived.read_bytes()).hexdigest()}
    (out/'VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    if not ok:raise SystemExit(1)

if __name__=='__main__':main()
