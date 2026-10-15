"""Portable source-based reconstruction of the paper's Fisher Top-K diagnostic.

This *reaggregates archived* paired seed rows and Jaccard pairs. It does not
retrain models or claim algorithm-level reproduction.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCALES={'madelon':(10,25,50), 'arcene':(10,500,1000)}


def reaggregate_reliability(root:Path, dest:Path) -> dict:
    p=root/'evidence/public/P0_72_81_AND_MECHANISM/P0_3/fisher_stability_pairwise.csv'
    r=root/'evidence/revision/v6_heldout_reliability/reconstructed_heldout_runs.csv'
    pairs=pd.read_csv(p)
    runs=pd.read_csv(r)
    if len(runs)!=40 or runs.duplicated(['dataset','seed','method']).any():
        raise ValueError('Expected 40 unique historical algorithm replay rows')
    required={'dataset','seed','method','train_hv_replayed','test_pf_hv_replayed','train_hv_abs_error'}
    if not required.issubset(runs.columns):
        raise ValueError(f'Missing required historical replay fields: {required-set(runs.columns)}')
    if runs.train_hv_abs_error.max()>1e-12 or runs.test_pf_hv_replayed.isna().any():
        raise ValueError('Archived replay did not validate against train HV, or test HV missing')
    rec=[]
    for ds,ks in SCALES.items():
        q=pairs[(pairs.dataset==ds)&(pairs.metric=='topk_jaccard')&(pairs.k.isin(ks))]
        sub=runs[runs.dataset==ds]
        for seed in range(72,82):
            ss=q[(q.seed_a==seed)|(q.seed_b==seed)]
            if len(ss)!=27 or set(ss.k)!=set(ks):
                raise ValueError(f'Missing reliability pair/scale values for {ds} seed {seed}: {len(ss)}')
            vals=sub[sub.seed==seed].set_index('method')
            if set(vals.index)!={'MSF-MOFS','Sparse-control'}:
                raise ValueError(f'Missing MSF/Sparse matched replay for {ds} seed {seed}')
            d={'dataset':ds,'seed':seed,'reliability_topk_jaccard':float(ss.groupby('k').value.mean().mean())}
            for label,method in [('msf','MSF-MOFS'),('sparse','Sparse-control')]:
                d[f'hv_train_{label}']=float(vals.loc[method,'train_hv_replayed'])
                d[f'hv_test_{label}']=float(vals.loc[method,'test_pf_hv_replayed'])
            d['delta_train']=d['hv_train_msf']-d['hv_train_sparse']
            d['delta_heldout']=d['hv_test_msf']-d['hv_test_sparse']
            rec.append(d)
    df=pd.DataFrame(rec)
    result=[]
    for outcome in ['delta_train','delta_heldout']:
        for name,g in [('pooled',df),('madelon',df[df.dataset=='madelon']),('arcene',df[df.dataset=='arcene'])]:
            rho,prob=spearmanr(g.reliability_topk_jaccard,g[outcome])
            result.append(dict(outcome=outcome,group=name,n=len(g),rho=float(rho),p_two_sided=float(prob),
                               mean_delta=float(g[outcome].mean()),min_delta=float(g[outcome].min()),max_delta=float(g[outcome].max()),
                               wins=int((g[outcome]>0).sum()), ties=int((g[outcome]==0).sum()),losses=int((g[outcome]<0).sum())))
    centered={}
    for outcome in ('delta_train','delta_heldout'):
        x=df.reliability_topk_jaccard-df.groupby('dataset').reliability_topk_jaccard.transform('mean')
        y=df[outcome]-df.groupby('dataset')[outcome].transform('mean')
        rho,prob=spearmanr(x,y)
        centered[outcome]={'rho':float(rho),'p_two_sided':float(prob)}
    dest.mkdir(parents=True,exist_ok=True)
    df.to_csv(dest/'reliability_seedlevel.csv',index=False,float_format='%.17g')
    summary=pd.DataFrame(result)
    summary.to_csv(dest/'reliability_association_summary.csv',index=False,float_format='%.17g')
    archived=pd.read_csv(root/'evidence/revision/v6_heldout_reliability/reliability_association_summary.csv')
    compare=summary.merge(archived,on=['outcome','group'],suffixes=('_new','_old'))
    errors=[]
    if len(compare)!=6:errors.append('Not all six archive summary rows present')
    for col in ['rho','p_two_sided','mean_delta']:
        if (compare[col+'_new']-compare[col+'_old']).abs().max()>1e-10:
            errors.append(f'{col} mismatch to archived summary')
    manifest={'status':'MATCH_ARCHIVED_REAGGREGATION' if not errors else 'MISMATCH',
              'evidence_level':'reaggregate_heldout_replay_and_jaccard_pairs_not_reexecute_search',
              'seed_rows':len(df),'summary_rows':len(summary),'dataset_centered_spearman':centered,
              'errors':errors,'sources':{str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [p,r]}}
    (dest/'PROVENANCE.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return manifest
