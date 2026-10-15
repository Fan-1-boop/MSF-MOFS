"""Independent, input-traceable Table 3 and HV-reference analyses.

Design constraint: this module NEVER reads the archived final summary as input.
It operates strictly on raw per-seed result rows and emits a provenance manifest.
The published 20,000-draw bootstrap seed/ordering is kept for complete 9-dataset runs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

CORE = ('madelon', 'hill_valley', 'musk1', 'arcene')
BREADTH = ('ALLAML', 'colon', 'GLI-85', 'Prostate-GE', 'SMK-CAN-187')
CARRIERS = ('NSGA-II', 'SPEA2')
INITIALIZERS = ('MSF', 'Random')
REFS = ((1., 1.), (1.05, 1.05), (1.10, 1.10), (1.10, 1.05), (1.05, 1.10))
REQUIRED = {'dataset', 'seed', 'carrier', 'init', 'test_pf_hv', 'best_test_mean_gmean_on_pf', 'min_features_on_pf', 'n_features'}


def holm(values):
    p = np.asarray(values, dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Invalid p-values')
    order = np.argsort(p, kind='stable')
    adjusted = np.empty_like(p)
    last = 0.
    for j, i in enumerate(order):
        last = max(last, float((len(p)-j)*p[i]))
        adjusted[i] = min(1., last)
    return adjusted


def rank_biserial_paired(d):
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    if not len(d):
        return 0.
    ranks = rankdata(np.abs(d), method='average')
    positive = ranks[d > 0].sum()
    negative = ranks[d < 0].sum()
    return float((positive-negative)/(positive+negative))


def _paired(group, value):
    piv = group.pivot(index='seed', columns='init', values=value)
    # pivot preserves deterministic ascending seed order.
    if piv[INITIALIZERS[0]].isna().any() or piv[INITIALIZERS[1]].isna().any():
        raise ValueError('Incomplete MSF/Random paired experiment')
    return piv, (piv['MSF'] - piv['Random']).to_numpy(dtype=float)


def _paired_row(dataset, carrier, group, value, *, rng=None, bootstrap_draws=20000):
    p, d = _paired(group, value)
    if not np.isfinite(d).all():
        raise ValueError(f'{dataset} {carrier}: invalid paired HV')
    try:
        prob = float(wilcoxon(d, alternative='two-sided', zero_method='wilcox').pvalue)
    except ValueError:
        prob = 1.
    row = dict(dataset=dataset, carrier=carrier, n=len(d),
               random_mean=float(p['Random'].mean()), msf_mean=float(p['MSF'].mean()),
               mean_delta=float(d.mean()), median_delta=float(np.median(d)),
               wins=int((d > 1e-12).sum()), ties=int((np.abs(d) <= 1e-12).sum()),
               losses=int((d < -1e-12).sum()), p_raw=prob, rbc=rank_biserial_paired(d))
    if rng is not None:
        # Equivalent RNG sequence to the original loop calling rng.choice(d, size=n)
        means = rng.choice(d, size=(bootstrap_draws, len(d)), replace=True).mean(axis=1)
        row['ci95_low'], row['ci95_high'] = [float(q) for q in np.quantile(means, [0.025, 0.975])]
    return row


def verify_input(frame, *, expected_datasets, expected_seeds=tuple(range(92,102))):
    missing = REQUIRED.difference(frame.columns)
    if missing:
        raise ValueError(f'Raw inputs missing columns: {sorted(missing)}')
    df = frame.copy()
    if len(df) == 0:
        raise ValueError('Empty raw input')
    # Canonicalize only the old public dataset's name; never relabel methods/carriers.
    df['dataset'] = df['dataset'].replace({'musk':'musk1', 'Musk1':'musk1'})
    dup = df.duplicated(['dataset','seed','carrier','init'])
    if dup.any():
        raise ValueError('Duplicate dataset/seed/carrier/init rows; source data mixed or resume error')
    dsets = set(df.dataset)
    if dsets != set(expected_datasets):
        raise ValueError(f'Dataset coverage mismatch: wanted={sorted(expected_datasets)}, got={sorted(dsets)}')
    expected = {(ds, sd, ca, it) for ds in expected_datasets for sd in expected_seeds for ca in CARRIERS for it in INITIALIZERS}
    got = {(r.dataset, int(r.seed), r.carrier, r.init) for r in df.itertuples(index=False)}
    if expected != got:
        raise ValueError(f'Missing experiment cells={len(expected-got)}, unexpected cells={len(got-expected)}; '
                         f'first missing={sorted(expected-got)[:3]}; first unexpected={sorted(got-expected)[:3]}')
    metrics = ['test_pf_hv','best_test_mean_gmean_on_pf','min_features_on_pf','n_features']
    if df[metrics].isna().any().any() or not np.isfinite(df[metrics].to_numpy(dtype=float)).all():
        raise ValueError('NaN/non-finite held-out objective fields')
    if not df['test_pf_hv'].between(0, 1+1e-10).all():
        raise ValueError('Base HV must be within [0,1] for ref=(1,1)')
    if not df['best_test_mean_gmean_on_pf'].between(0, 1+1e-10).all():
        raise ValueError('Best held-out G-mean invalid')
    if not ((df['min_features_on_pf'] >= 0) & (df['min_features_on_pf'] <= df['n_features']) & (df['n_features'] > 0)).all():
        raise ValueError('Feature counts invalid')
    return df


def build_tables(frame, *, bootstrap_seed=4302026, bootstrap_draws=20000):
    """Precondition: all requested dataset/carrier/init/seed cells are present.

    The seeded bootstrap is identical to original calculation ONLY when the frame
    contains all nine datasets sorted by dataset/carrier. Breadth-only CIs are
    valid CIs, but do not inherit the old global generator state.
    """
    rng=np.random.default_rng(bootstrap_seed)
    ci=[]
    for (ds,ca),g in frame.groupby(['dataset','carrier'],sort=True):
        ci.append(_paired_row(ds,ca,g,'test_pf_hv',rng=rng,bootstrap_draws=bootstrap_draws))
    ci=pd.DataFrame(ci)
    ci['p_holm_within_carrier']=np.nan
    for ca,ids in ci.groupby('carrier').groups.items():
        ci.loc[ids,'p_holm_within_carrier']=holm(ci.loc[ids,'p_raw'])
    sens=[]
    for r1,r2 in REFS:
        x=frame.copy()
        # Exact 2D minimization HV expansion for front entirely below (1,1).
        # min f1 = 1-max(Gmean), min f2 = min(subset-size ratio).
        a,b=r1-1.,r2-1.
        f1min=1-x['best_test_mean_gmean_on_pf']
        f2min=x['min_features_on_pf']/x['n_features']
        x['hv_ref']=x['test_pf_hv']+a*(1-f2min)+b*(1-f1min)+a*b
        for (ds,ca),g in x.groupby(['dataset','carrier'],sort=True):
            item=_paired_row(ds,ca,g,'hv_ref')
            item['ref']=f'({r1:.2f},{r2:.2f})'
            sens.append(item)
    sens=pd.DataFrame(sens)
    sens['p_holm_within_ref_carrier']=np.nan
    for _, ids in sens.groupby(['ref','carrier']).groups.items():
        sens.loc[ids,'p_holm_within_ref_carrier']=holm(sens.loc[ids,'p_raw'])
    summary=[]
    for (ref,ca),g in sens.groupby(['ref','carrier']):
        summary.append(dict(ref=ref,carrier=ca,datasets=len(g),
                            positive_mean_delta=int((g.mean_delta>0).sum()),
                            negative_mean_delta=int((g.mean_delta<0).sum()),
                            holm_sig_positive=int(((g.mean_delta>0)&(g.p_holm_within_ref_carrier<.05)).sum()),
                            holm_sig_negative=int(((g.mean_delta<0)&(g.p_holm_within_ref_carrier<.05)).sum()),
                            total_seed_wins=int(g.wins.sum()),total_seed_ties=int(g.ties.sum()),
                            total_seed_losses=int(g.losses.sum()),mean_dataset_delta=float(g.mean_delta.mean())))
    return ci, sens, pd.DataFrame(summary)


def analyze_files(inputs: list[Path], dest: Path, *, scope='nine', expected_seeds=tuple(range(92,102)),
                  evidence_type='reproduced', bootstrap_draws=20000):
    """Analyze complete requested scope only; reject incomplete rather than silently drop cells."""
    if scope not in ('nine', 'core', 'breadth'):
        raise ValueError('Scope must be nine/core/breadth')
    expected = CORE + BREADTH if scope == 'nine' else CORE if scope == 'core' else BREADTH
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    # Dataset-specific file provenance must be declared honestly (archive vs new runs).
    project_root=Path(__file__).resolve().parents[2]
    archive_dirs=[(project_root/'evidence').resolve(), (project_root/'results/reference').resolve()]
    archive_flags=[any(p.resolve().is_relative_to(folder) for folder in archive_dirs) for p in inputs]
    if evidence_type=='reproduced' and any(archive_flags):
        raise ValueError('Archived CSV is not a reproduced algorithm run; use --evidence-type mixed')
    if evidence_type=='archived_per_seed' and not all(archive_flags):
        raise ValueError('Raw output is not archived evidence; declare reproduced or mixed')
    frames=[pd.read_csv(p) for p in inputs]
    raw=verify_input(pd.concat(frames,ignore_index=True),expected_datasets=expected,expected_seeds=expected_seeds)
    ci,sens,summ=build_tables(raw,bootstrap_draws=bootstrap_draws)
    dest.mkdir(parents=True,exist_ok=True)
    ci.to_csv(dest/'paired_hv_ci_effect.csv',index=False,float_format='%.17g')
    sens.to_csv(dest/'hv_reference_sensitivity.csv',index=False,float_format='%.17g')
    summ.to_csv(dest/'hv_reference_sensitivity_summary.csv',index=False,float_format='%.17g')
    manifest={
        'status':'REAGGREGATED_FROM_RAW', 'not_full_end_to_end_claim':True,
        'full_public_table3_search_records_present':scope=='nine' and evidence_type=='reproduced',
        'scope':scope,'evidence_type':evidence_type,
        'expected_seeds':list(expected_seeds),'bootstrap_seed':4302026,'bootstrap_draws':bootstrap_draws,
        'groups':len(ci),'raw_rows':len(raw),
        'input_files':[{'path':(p.resolve().relative_to(project_root).as_posix() if p.resolve().is_relative_to(project_root) else None),
                        'file_name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(), 'rows':len(f)}
                       for p,f in zip(inputs,frames)],
        'numeric_checks':{'per_carrier_wins':{ca:int(g.wins.sum()) for ca,g in ci.groupby('carrier')},
                          'per_carrier_holm_significant':{ca:int((g.p_holm_within_carrier<.05).sum())
                                                          for ca,g in ci.groupby('carrier')}}
    }
    if scope=='nine':
        # Validation only: the archived *summary* is NEVER fed into the estimator.
        reference_path=project_root/'evidence/revision/statistics_existing_results/paired_hv_ci_effect_9dataset.csv'
        if reference_path.is_file():
            ref=pd.read_csv(reference_path)
            merged=ci.merge(ref,on=['dataset','carrier'],suffixes=('_new','_archive'),how='outer',indicator=True)
            errors=[]
            if len(merged)!=18 or (merged['_merge']!='both').any():
                errors.append('Unexpected archived Table 3 grouping')
            diffs={}
            for col in ['mean_delta','ci95_low','ci95_high','p_raw','rbc','p_holm_within_carrier']:
                diffs[col]=float((merged[col+'_new']-merged[col+'_archive']).abs().max())
            for col in ['n','wins','ties','losses']:
                mismatched=bool((merged[col+'_new']!=merged[col+'_archive']).any())
                if mismatched: errors.append(f'{col}: archived paired count mismatch')
            if diffs['mean_delta']>1e-9:
                errors.append('The new central HV differences do not match archived statistics')
            if diffs['p_raw']>1e-10 or diffs['p_holm_within_carrier']>1e-10:
                errors.append('The new Wilcoxon/Holm test statistics do not match archived statistics')
            if diffs['ci95_low']>1e-9 or diffs['ci95_high']>1e-9:
                errors.append('Bootstrap CIs differ from archived output; inspect raw inputs and RNG order')
            compare={'status':'MATCH' if not errors else 'MISMATCH','compared_groups':len(merged),
                     'max_absolute_differences':diffs,'errors':errors,
                     'note':'Independent results compared against archive; archive summary not used as a computational input'}
            (dest/'PAPER_TABLE3_COMPARISON.json').write_text(
                json.dumps(compare,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
            manifest['archive_summary_comparison']=compare['status']
    (dest/'PROVENANCE.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return manifest
