"""v4 scope-aware status and independent Fisher reliability association checks."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from msf_repro.validation_v3 import verify_project
from msf_repro.coverage import report_coverage

SCALES={'madelon':(10,25,50),'arcene':(10,500,1000)}


def reliability_from_reruns(root:Path,out:Path)->dict:
    generated=root/'results/reference/reliability/replayed_runs.csv'
    fisher=root/'results/reference/fisher_stability/pairwise.csv'
    historic=root/'evidence/revision/v6_heldout_reliability/reliability_association_summary.csv'
    report={'complete':False,'reference_status':'PENDING','expected_search_rows':40,'archived_replay_rows':0, 'new_search_rows':0}
    if not generated.exists() or not fisher.exists():return report
    runs=pd.read_csv(generated)
    keys=['dataset','seed','method']
    expect={(ds,s,m) for ds in SCALES for s in range(72,82) for m in ('MSF-MOFS','Sparse-control')}
    got={(str(r.dataset),int(r.seed),str(r.method)) for r in runs.itertuples(index=False)}
    report['new_search_rows']=len(expect&got)
    if len(got)!=len(runs) or expect!=got:return report
    pairs=pd.read_csv(fisher)
    rows=[]
    for ds,ks in SCALES.items():
        data=pairs[(pairs.dataset==ds)&(pairs.metric=='topk_jaccard')&(pairs.k.isin(ks))]
        for seed in range(72,82):
            subset=data[(data.seed_a==seed)|(data.seed_b==seed)]
            if len(subset)!=27 or set(subset.k.astype(int))!=set(ks):
                report['reference_status']='INVALID_FISHER_PAIRS';return report
            run=runs[(runs.dataset==ds)&(runs.seed==seed)].set_index('method')
            if len(run)!=2: return report
            row={'dataset':ds,'seed':seed,'reliability_topk_jaccard':float(subset.groupby('k').value.mean().mean())}
            row['delta_train']=float(run.loc['MSF-MOFS','train_hv_replayed']-run.loc['Sparse-control','train_hv_replayed'])
            row['delta_heldout']=float(run.loc['MSF-MOFS','test_pf_hv_replayed']-run.loc['Sparse-control','test_pf_hv_replayed'])
            rows.append(row)
    rec=pd.DataFrame(rows)
    corr=[]
    for outcome in ('delta_train','delta_heldout'):
        for name,g in [('pooled',rec),('madelon',rec[rec.dataset=='madelon']),('arcene',rec[rec.dataset=='arcene'])]:
            rho,p=spearmanr(g.reliability_topk_jaccard,g[outcome])
            corr.append({'outcome':outcome,'group':name,'n':len(g),'rho':float(rho),'p_two_sided':float(p)})
    summary=pd.DataFrame(corr)
    old=pd.read_csv(historic)
    merged=summary.merge(old,on=['outcome','group'],validate='one_to_one',suffixes=('_new','_paper'))
    if len(merged)!=6:raise RuntimeError('Missing six historical reliability contrasts')
    diff=max(float(np.max(np.abs(merged[f'{metric}_new']-merged[f'{metric}_paper']))) for metric in ('rho','p_two_sided'))
    rec.to_csv(out/'reliability_rerun_seedlevel.csv',index=False,float_format='%.17g')
    merged.to_csv(out/'RELIABILITY_PAPER_COMPARISON.csv',index=False,float_format='%.17g')
    report.update(complete=True,reference_status=('MATCH' if diff<1e-12 else 'MISMATCH'),
        max_abs_rho_p_diff=diff,n_seed_level_pairs=len(rec),n_association_statistics=len(merged),
        new_replay_sha256=hashlib.sha256(generated.read_bytes()).hexdigest())
    return report


def v4_status(root:Path)->dict:
    root=root.resolve();out=root/'results/reproduced/paper/verification/v4';out.mkdir(parents=True,exist_ok=True)
    v3=verify_project(root)
    pub=report_coverage(root,root/'results/reproduced/paper/audit/run_coverage.json')
    fishfile=root/'results/reference/fisher_stability/verification.json'
    fisher=json.loads(fishfile.read_text(encoding='utf-8')) if fishfile.exists() else {'status':'NOT_RUN'}
    replay=reliability_from_reruns(root,out)
    checks={}
    for name in ('TABLE2_PAPER_CHECK','POP50_PAPER_CHECK','TABLE4_PAPER_CHECK'):
        p=root/f'results/reproduced/paper/verification/v3/{name}.csv'
        checks[name]={'exists':p.exists(),'all_4dp_match':bool(pd.read_csv(p).matches_4dp.all()) if p.exists() else False}
    result={'table3_public_360':pub['full_public_search_complete'],
        'table2_external_120':v3['status_summary']['full_table2'],
        'sparse_control_50':v3['status_summary']['full_sparse'],
        'population50_60':v3['status_summary']['full_pop50'],
        'satellite_40_pairs':v3['status_summary']['full_satellite'],
        'fisher_stability_450':fisher,'reliability_40_searches':replay,
        'paper_rounded_checks':checks,
        'paper_wide_end_to_end_reproduced':False,
        'explicit_limitations':['V8 original scenario-generation dependencies unavailable',
          'V6 frozen 140-feature model and timing harness now available; historical 59.8% timing depends on environment',
          'Timing metrics not directly comparable between CPU environments',
          'SPEA2 source provenance includes reconstructed source, not bytewise original']}
    (out/'V4_EVIDENCE_STATUS.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return result

if __name__=='__main__':
    root=Path(__file__).resolve().parents[2];print(json.dumps(v4_status(root),indent=2,ensure_ascii=False))
