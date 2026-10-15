"""Compare one RE-RUN against available per-seed historical evidence.

Comparison by actual method/dataset/seed; never reinterpret summary as full raw evidence.
"""
from pathlib import Path
import pandas as pd


def compare_demo_with_archived_core(root: Path, run_csv: Path, dataset: str, seed: int) -> dict:
    a=pd.read_csv(run_csv)
    # Archived external comparison has MSF-NSGA-II but NOT the original core carrier Random/SPEA2 rows.
    reference=pd.read_csv(root/'evidence/public/core_external_confirmation/confirmation_raw.csv')
    historical='musk' if dataset=='musk1' else dataset
    r=reference[(reference.dataset==historical)&(reference.seed==seed)&(reference.method=='MSF-MOFS')]
    m=a[(a.dataset==dataset)&(a.seed==seed)&(a.carrier=='NSGA-II')&(a.init=='MSF')]
    if len(r)!=1 or len(m)!=1:
        return {'status':'NO_MATCH','source':'core_external_confirmation/confirmation_raw.csv',
                'note':'No matching per-seed MSF-NSGA-II row for this dataset/seed.'}
    fields=['train_hv','test_pf_hv','unique_wrapper_evaluations']
    pairs={k: {'reference':float(r.iloc[0][k]),'rerun':float(m.iloc[0][k])} for k in fields}
    equal=all(abs(v['reference']-v['rerun'])<1e-10 for k,v in pairs.items())
    return {'status':'MATCH' if equal else 'MISMATCH','fields':pairs,
            'verified':'single dataset-seed MSF NSGA-II only',
            'unverified':'the other three demo rows are not matched to complete core original per-seed evidence'}


def compare_satellite_with_archived(root:Path, run_csv:Path, subsystem:str) -> dict:
    a=pd.read_csv(run_csv)
    r=pd.read_csv(root/f'evidence/satellite/raw_per_seed/{subsystem}/per_seed_results.csv')
    merged=a.merge(r[['seed','msf_hv','random_hv']],on='seed',suffixes=('_new','_old'))
    mismatches=[]
    # Satellite reference rounded to 6 decimals.
    for _,row in merged.iterrows():
        for field in ('msf_hv','random_hv'):
            if abs(row[f'{field}_new']-row[f'{field}_old'])>0.00000051:
                mismatches.append(f"seed={int(row['seed'])}, {field}")
    if len(merged)!=len(a):mismatches.append('Unmatched seed(s)')
    return {'status':'MATCH' if not mismatches else 'MISMATCH',
            'checked_seeds':len(merged),'comparison_precision':'archived values rounded to 6 decimal places',
            'mismatches':mismatches}


def compare_breadth_full_with_archive(root:Path, new_csv:Path) -> dict:
    """Independent per-seed regression test, requires all 200 breadth search cells.

    Timing fields are intentionally ignored: hardware/load-dependent.
    """
    new=pd.read_csv(new_csv)
    old=pd.read_csv(root/'evidence/public/untouched_breadth/raw.csv')
    keys=['dataset','seed','carrier','init']
    fields=['train_hv','test_pf_hv','pf_size','best_test_mean_gmean_on_pf',
            'best_test_worst_gmean_on_pf','min_features_on_pf','unique_wrapper_evaluations']
    if len(new)!=200 or len(old)!=200 or new.duplicated(keys).any() or old.duplicated(keys).any():
        return {'status':'INVALID_COVERAGE','expected_rows':200,'new_rows':len(new),'old_rows':len(old)}
    merged=new.merge(old,on=keys,suffixes=('_new','_archive'),how='outer',indicator=True,validate='one_to_one')
    if len(merged)!=200 or (merged._merge!='both').any():
        return {'status':'INVALID_COVERAGE','note':'Mismatch in dataset/seed/carrier/init keys'}
    maxdiff={}
    mismatches=[]
    for field in fields:
        a=merged[field+'_new'].astype(float)
        b=merged[field+'_archive'].astype(float)
        diff=(a-b).abs()
        maxdiff[field]=float(diff.max())
        if diff.max()>1e-10:
            mismatches.append(f'{field}: max |new - archived| = {diff.max():.12g}')
    return {'status':'MATCH' if not mismatches else 'MISMATCH',
            'compared_rows':200,'numerical_comparison_tolerance':1e-10,
            'max_abs_difference':maxdiff,'mismatches':mismatches,
            'note':'True rerun vs archived per-seed regression; not an independent replication on different hardware.'}


def compare_core_msf_nsga_with_archive(root:Path, new_csv:Path) -> dict:
    """Only the externally archived MSF/NSGA-II 40 rows, not SPEA2/Random."""
    new=pd.read_csv(new_csv)
    new=new[(new.carrier=='NSGA-II')&(new['init']=='MSF')].copy()
    reference=pd.read_csv(root/'evidence/public/core_external_confirmation/confirmation_raw.csv')
    reference=reference[reference.method=='MSF-MOFS'].copy()
    reference['dataset']=reference['dataset'].replace({'musk':'musk1'})
    keys=['dataset','seed']
    fields=['train_hv','test_pf_hv','unique_wrapper_evaluations']
    if len(new)!=40 or len(reference)!=40 or new.duplicated(keys).any() or reference.duplicated(keys).any():
        return {'status':'INVALID_COVERAGE','expected_rows':40,'new_rows':len(new),'archived_rows':len(reference)}
    merged=new.merge(reference,on=keys,suffixes=('_new','_archive'),validate='one_to_one',how='outer',indicator=True)
    if len(merged)!=40 or (merged._merge!='both').any():
        return {'status':'INVALID_COVERAGE','note':'MSF-NSGA dataset/seed mismatch'}
    differences={col:float((merged[col+'_new'].astype(float)-merged[col+'_archive'].astype(float)).abs().max())
                 for col in fields}
    failed={k:v for k,v in differences.items() if v>1e-10}
    return {'status':'MATCH' if not failed else 'MISMATCH','compared_rows':40,
            'max_abs_difference':differences,'mismatches':failed,
            'verified':'MSF+NSGA-II only; original external-comparison per-seed rows',
            'not_verified':'Random and SPEA2 core results lack full original per-seed primary evidence; all four methods were newly executed.'}
