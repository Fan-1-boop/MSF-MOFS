"""Evidence-aware verification for MSF-MOFS v3.

Never promotes archived statistical reaggregation to an independently rerun search.
All completeness tests are strict and keyed by exact datasets / seeds / methods.
Wall-clock time is reported separately and is not used for numerical equivalence.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata
from msf_repro.paper_table_checks import check_paper_tables

CORE=['madelon','hill_valley','musk1','arcene']
BREADTH=['ALLAML','colon','GLI-85','Prostate-GE','SMK-CAN-187']
SEEDS=list(range(92,102))
SEEDS_POP=list(range(72,82))
SUBSYSTEMS=['power','thermal','attitude','data']
EXTERNAL=['MSF-MOFS','HFMOEA-CP','MBLPE-CP']
NUMERIC_EXT=['train_hv','test_pf_hv','pf_size','best_test_mean_gmean_on_pf_diagnostic','best_test_worst_gmean_on_pf_diagnostic','min_features_on_pf','unique_wrapper_evaluations']
NUMERIC_SPARSE=['train_hv','test_pf_hv','pf_size','best_test_mean_gmean_on_pf','best_test_worst_gmean_on_pf','min_features_on_pf','unique_wrapper_evaluations']
NUMERIC_POP=['gen0_test_pf_hv','gen1_test_pf_hv','gen0_pf_size','gen1_pf_size','gen0_unique_wrapper_evaluations','gen1_unique_wrapper_evaluations','gen1_train_hv']
NUMERIC_SAT=['msf_hv','random_hv','msf_test_gmean','random_test_gmean','msf_feature_count','random_feature_count']


def validate_rows(path:Path, keys:list[str], expected:set[tuple], reference:Path|None=None,
                  numeric:list[str]|None=None, tol:float=1e-11)->tuple[dict,pd.DataFrame|None]:
    report={'path':str(path),'expected':len(expected),'completed':0,'missing':len(expected),'extra':0,'duplicates':0,
            'complete':False,'reference_status':'NOT_CHECKED'}
    if not path.exists():
        report['reason']='MISSING_RESULT_FILE'
        return report,None
    df=pd.read_csv(path)
    if any(k not in df.columns for k in keys):
        report['reason']='MISSING_KEY_COLUMNS'
        return report,None
    # Keys represented consistently across float-int CSV encoding.
    kt=[tuple(int(r[k]) if k=='seed' else str(r[k]) for k in keys) for _,r in df.iterrows()]
    found=set(kt)
    report.update(completed=len(found & expected),missing=len(expected-found),extra=len(found-expected),duplicates=len(kt)-len(found))
    report['complete']=report['missing']==report['extra']==report['duplicates']==0
    report['missing_examples']=[list(k) for k in sorted(expected-found)[:5]]
    if reference is None or not reference.exists():return report,df
    ref=pd.read_csv(reference)
    # Archived external baseline runner names UCI Musk1 as 'musk'.
    # Formal v3 loader uses 'musk1'; only normalize this documented alias.
    if reference.name=='confirmation_raw.csv' and 'dataset' in ref.columns and 'method' in ref.columns:
        ref['dataset']=ref['dataset'].replace({'musk':'musk1'})
    if any(k not in ref.columns for k in keys):
        report['reference_status']='INVALID_REFERENCE_KEYS'
        return report,df
    try:
        a=df.set_index(keys,verify_integrity=True)
        b=ref.set_index(keys,verify_integrity=True)
    except ValueError:
        report['reference_status']='DUPLICATE_KEYS_IN_ONE_OR_BOTH'
        return report,df
    both=a.index.intersection(b.index)
    if not len(both):
        report['reference_status']='NO_COMMON_KEYS'
        return report,df
    columns=list(numeric or [])
    if any(k not in a.columns or k not in b.columns for k in columns):
        report['reference_status']='MISSING_REFERENCE_METRICS'
        return report,df
    errors={}
    for k in columns:
        local_tol=(5e-5 if 'gmean' in k.lower() else tol)
        va=pd.to_numeric(a.loc[both,k],errors='coerce').to_numpy(float)
        vb=pd.to_numeric(b.loc[both,k],errors='coerce').to_numpy(float)
        err=np.abs(va-vb)
        is_equal=np.isclose(va,vb,atol=local_tol,rtol=0.,equal_nan=True)
        errors[k]={'max_abs_diff':float(np.nanmax(err)) if len(err) else 0.,'mismatches':int((~is_equal).sum())}
    report.update(reference_rows_compared=len(both),comparison_metrics=errors,
                  reference_status=('MATCH' if all(x['mismatches']==0 for x in errors.values()) else 'MISMATCH'))
    return report,df


def holm(pvals):
    p=np.asarray(pvals,float); idx=np.argsort(p,kind='stable'); out=np.empty(len(p))
    old=0.
    for i,k in enumerate(idx):
        old=max(old,(len(p)-i)*p[k]);out[k]=min(1.,old)
    return out


def rb(d):
    d=np.asarray(d,float);d=d[~np.isclose(d,0.,atol=1e-15,rtol=0)]
    if len(d)==0:return 0.
    ranks=rankdata(abs(d),method='average')
    return float((ranks[d>0].sum()-ranks[d<0].sum())/ranks.sum())


def pair_stats(left,right,dataset,contrast,stage=None):
    a=left.set_index('seed')['test_pf_hv']
    b=right.set_index('seed')['test_pf_hv']
    if set(a.index)!=set(b.index) or len(a)!=10 or len(a.index.unique())!=10:
        raise ValueError(f'Incomplete paired seeds: {dataset}/{contrast}')
    d=(a.sort_index()-b.sort_index()).to_numpy(float)
    p=1. if np.allclose(d,0,atol=1e-15,rtol=0) else float(wilcoxon(d,zero_method='wilcox',alternative='two-sided',method='auto').pvalue)
    out={'dataset':dataset,'contrast':contrast,'n':len(d),'mean_delta_hv':float(d.mean()),
         'wins':int((d>1e-15).sum()),'ties':int((np.abs(d)<=1e-15).sum()),
         'losses':int((d< -1e-15).sum()),'rbc':rb(d),'p_raw':p}
    if stage:out['stage']=stage
    return out


def external_stats(df:pd.DataFrame,out:Path):
    if not all(((df.dataset==ds)&(df.method==meth)).sum()==10 for ds in CORE for meth in EXTERNAL):
        raise ValueError('External results are not 4 datasets x 3 methods x 10 seeds')
    summary=df.groupby(['dataset','method'],sort=False).agg(
        mean_hv=('test_pf_hv','mean'),std_hv=('test_pf_hv','std'),
        mean_prep_time=('ranking_or_filter_time','mean'),mean_total_time=('total_time','mean')).reset_index()
    summary.to_csv(out/'table2_recomputed.csv',index=False)
    rows=[]
    for base in EXTERNAL[1:]:
        family=[]
        for ds in CORE:
            g=df[df.dataset==ds]
            family.append(pair_stats(g[g.method=='MSF-MOFS'],g[g.method==base],ds,f'MSF-MOFS - {base}'))
        # Per paper: Holm family consists of four core datasets, PER external baseline.
        for r,p in zip(family,holm([x['p_raw'] for x in family])):r['holm_p_4_datasets_per_baseline']=float(p)
        rows.extend(family)
    pd.DataFrame(rows).to_csv(out/'external_paired_paper_holm.csv',index=False)
    # Rank methods by each dataset's held-out mean HV (not per seed), then average over datasets.
    ranks=[]
    for ds in CORE:
        g=summary[summary.dataset==ds].set_index('method')
        rs=rankdata(-g.loc[EXTERNAL,'mean_hv'].to_numpy(float),method='average')
        for meth,r in zip(EXTERNAL,rs):ranks.append({'dataset':ds,'method':meth,'rank':float(r)})
    pd.DataFrame(ranks).to_csv(out/'table2_dataset_ranks.csv',index=False)
    pd.DataFrame(ranks).groupby('method')['rank'].mean().reset_index(name='average_rank').to_csv(out/'table2_average_ranks.csv',index=False)
    return summary


def sparse_stats(sparse:pd.DataFrame,pub:pd.DataFrame,out:Path):
    base=pub[(pub.carrier=='NSGA-II') & (pub.dataset.isin(BREADTH)) & (pub['init'].isin(['MSF','Random']))].copy()
    selected=sparse[['dataset','seed','carrier','init','test_pf_hv']]
    whole=pd.concat([base,selected],ignore_index=True)
    for ds in BREADTH:
        for method in ['MSF','Random','Sparse-control']:
            if len(whole[(whole.dataset==ds)&(whole['init']==method)])!=10:
                raise ValueError(f'Incomplete sparse/Random/MSF cell {ds}/{method}')
    contrasts=[('Sparse-control','Random'),('MSF','Sparse-control'),('MSF','Random')]
    rows=[]
    for a,b in contrasts:
        fam=[]
        for ds in BREADTH:
            g=whole[whole.dataset==ds]
            fam.append(pair_stats(g[g['init']==a],g[g['init']==b],ds,f'{a} - {b}'))
        for r,q in zip(fam,holm([x['p_raw'] for x in fam])):r['holm_p_5_datasets_per_contrast']=float(q)
        rows.extend(fam)
    pd.DataFrame(rows).to_csv(out/'sparse_paired_paper_holm.csv',index=False)
    whole.to_csv(out/'sparse_three_way_matched_raw.csv',index=False)


def pop_stats(parts:list[pd.DataFrame],out:Path):
    df=pd.concat(parts,ignore_index=True)
    expected={(ds,seed,m) for ds in ['madelon','hill_valley','Prostate-GE'] for seed in SEEDS_POP for m in ['MSF-MOFS','Random']}
    found={(str(r.dataset),int(r.seed),str(r.method)) for r in df.itertuples(index=False)}
    if expected!=found or len(df)!=len(found):raise ValueError('Population50 incomplete 60 unique cells')
    rows=[]
    for stage in ('gen0','gen1'):
        family=[]
        for ds in ['madelon','hill_valley','Prostate-GE']:
            g=df[df.dataset==ds]
            a=g[g.method=='MSF-MOFS'].set_index('seed')[f'{stage}_test_pf_hv']
            b=g[g.method=='Random'].set_index('seed')[f'{stage}_test_pf_hv']
            d=(a.sort_index()-b.sort_index()).to_numpy(float)
            p=float(wilcoxon(d,alternative='two-sided').pvalue)
            family.append({'dataset':ds,'stage':stage,'mean_delta_hv':float(d.mean()),'wins':int((d>1e-15).sum()),
                           'ties':int((np.abs(d)<=1e-15).sum()),'losses':int((d<-1e-15).sum()),'rbc':rb(d),'p_raw':p,
                           'guided_count':3,'population':50})
        for r,q in zip(family,holm([x['p_raw'] for x in family])):r['holm_p_3_datasets_per_stage']=float(q)
        rows.extend(family)
    pd.DataFrame(rows).to_csv(out/'population50_paired_paper_holm.csv',index=False)
    df.to_csv(out/'population50_3dataset_all_raw.csv',index=False)


def satellite_stats(recs:dict[str,pd.DataFrame],out:Path):
    rows=[]
    for name in SUBSYSTEMS:
        df=recs[name]
        d=(df.sort_values('seed').msf_hv-df.sort_values('seed').random_hv).to_numpy(float)
        rows.append({'subsystem':name,'n':len(d),'msf_mean':float(df.msf_hv.mean()),
                     'msf_std':float(df.msf_hv.std(ddof=1)),
                     'random_mean':float(df.random_hv.mean()),'random_std':float(df.random_hv.std(ddof=1)),
                     'mean_delta_hv':float(d.mean()),'wins':int((d>1e-15).sum()),
                     'ties':int((abs(d)<=1e-15).sum()),'losses':int((d<-1e-15).sum()),
                     'p_raw':float(wilcoxon(d,alternative='two-sided').pvalue),'rbc':rb(d)})
    for r,q in zip(rows,holm([r['p_raw'] for r in rows])):r['holm_p_4_subsystems']=float(q)
    pd.DataFrame(rows).to_csv(out/'table4_recomputed.csv',index=False)


def verify_project(root:Path,write_dir:Path|None=None):
    root=Path(root); out=Path(write_dir or root/'results/reproduced/paper/verification/v3');out.mkdir(parents=True,exist_ok=True)
    result={}
    ev=root/'evidence';current=root/'results/reference'
    ext,extdf=validate_rows(current/'table2_external/raw.csv',['dataset','seed','method'],
      {(ds,s,m) for ds in CORE for s in SEEDS for m in EXTERNAL},ev/'public/core_external_confirmation/confirmation_raw.csv',NUMERIC_EXT)
    result['table2_external']=ext
    if ext['complete'] and ext['reference_status']=='MATCH':
        external_stats(extdf,out)
        check_paper_tables(out,'table2').to_csv(out/'TABLE2_PAPER_CHECK.csv',index=False)
    sparse,sparsedf=validate_rows(current/'sparse_control/raw.csv',['dataset','seed','init'],
      {(ds,s,'Sparse-control') for ds in BREADTH for s in SEEDS},ev/'public/sparse_control_attribution/sparse_control_raw.csv',NUMERIC_SPARSE)
    result['sparse_control']=sparse
    if sparse['complete'] and sparse['reference_status']=='MATCH':
        sparse_stats(sparsedf,pd.read_csv(current/'table3/breadth/raw.csv'),out)
    poprows=[];popcomplete=True
    for label,path,ref,ds in [
       ('madelon',current/'population50/madelon/raw.csv',ev/'public/P0_72_81_AND_MECHANISM/P0_4/large_population_raw.csv',['madelon']),
       ('extension',current/'population50/extension/raw.csv',ev/'revision/population50_extension/large_population_extension_raw.csv',['hill_valley','Prostate-GE'])]:
        status,raw=validate_rows(path,['dataset','seed','method'],{(a,s,m) for a in ds for s in SEEDS_POP for m in ['MSF-MOFS','Random']},ref,NUMERIC_POP)
        result['population50_'+label]=status
        if status['complete'] and status['reference_status']=='MATCH':poprows.append(raw)
        else:popcomplete=False
    if popcomplete:
        pop_stats(poprows,out)
        check_paper_tables(out,'population50').to_csv(out/'POP50_PAPER_CHECK.csv',index=False)
    satrecs={}
    for name in SUBSYSTEMS:
        status,df=validate_rows(root/'results/reference/table4_satellite'/name/'per_seed_results.csv',['seed'],
             {(s,) for s in range(192,202)},ev/f'satellite/raw_per_seed/{name}/per_seed_results.csv',NUMERIC_SAT,tol=5e-7)
        result['satellite_'+name]=status
        if status['complete'] and status['reference_status']=='MATCH':satrecs[name]=df
    if len(satrecs)==4:
        satellite_stats(satrecs,out)
        check_paper_tables(out,'table4').to_csv(out/'TABLE4_PAPER_CHECK.csv',index=False)
    result['status_summary']={'full_table2':ext['complete'] and ext['reference_status']=='MATCH',
        'full_sparse':sparse['complete'] and sparse['reference_status']=='MATCH',
        'full_pop50':popcomplete,
        'full_satellite':len(satrecs)==4,
        'full_paper_reproduced':False, # frozen 140-feature deployment remains missing
        'scope_note':'140-feature extraction and frozen deployment artifacts still missing'}
    (out/'V3_EVIDENCE_STATUS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result
