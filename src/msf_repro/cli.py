"""Portable CLI for the frozen *existing* research implementations.

论文已有实验结果保存在 results/reference/。新搜索结果及本次审计写入
results/reproduced/paper/，以防止误覆盖已归档原始证据。
"""
from __future__ import annotations

import argparse
import json
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from msf_repro.audit import run_archive_audit, SUBSYSTEMS
from msf_repro.replay import compare_demo_with_archived_core, compare_satellite_with_archived, compare_breadth_full_with_archive, compare_core_msf_nsga_with_archive
from msf_repro.statistics import analyze_files
from msf_repro.reliability import reaggregate_reliability
from msf_repro.coverage import report_coverage
from msf_repro.merge import merge_core_inputs
from msf_repro.validation_v3 import verify_project
from msf_repro.validation_v4 import v4_status

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT/'data/public/core'
BREADTH = ROOT/'data/public/breadth'
OUT = ROOT/'results/reference'  # 已归档的历史记录：只读输入
LOCAL = ROOT/'results/reproduced/paper'  # 全部新实验和运行审计默认写到这里
PROTOCOL = json.loads((ROOT/'configs/frozen_protocol.json').read_text(encoding='utf-8'))
DEFAULT_SEEDS = PROTOCOL['confirmation_seeds']
POP = str(PROTOCOL['population'])
GENERATIONS = str(PROTOCOL['generations'])


def _run(path: Path, arguments: list[str], extra_paths: list[Path] | None = None):
    env=os.environ.copy()
    pp=[str(ROOT/'src')] + [str(p) for p in (extra_paths or [])]
    if env.get('PYTHONPATH'):pp.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = os.pathsep.join(pp)
    command=[sys.executable, str(path), *arguments]
    print('RUN:', ' '.join(command), flush=True)
    log = LOCAL/'invocations.jsonl'
    log.parent.mkdir(parents=True,exist_ok=True)
    entry={'utc':datetime.now(timezone.utc).isoformat(),'runner':str(path.relative_to(ROOT)),
           'runner_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
           'argv':arguments,'python':sys.version.split()[0]}
    with log.open('a',encoding='utf-8') as f:
        f.write(json.dumps({**entry,'event':'start'},ensure_ascii=False)+'\n')
    try:
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    except BaseException as exc:
        with log.open('a',encoding='utf-8') as f:
            f.write(json.dumps({**entry,'event':'error','message':str(exc)},ensure_ascii=False)+'\n')
        raise
    else:
        with log.open('a',encoding='utf-8') as f:
            f.write(json.dumps({**entry,'event':'success'},ensure_ascii=False)+'\n')


def main(argv=None):
    p=argparse.ArgumentParser(description='MSF-MOFS: frozen paper run/archived evidence audit')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('check',help='Recompute stored statistics and validate dataset hashes (not rerun search)')
    sub.add_parser('smoke',help='Run fast API tests without benchmark claims')
    d=sub.add_parser('demo',help='Real core dataset, 1 seed, 2 carriers x 2 initializers')
    d.add_argument('--dataset', choices=['madelon','hill_valley','musk1','arcene'],default='madelon')
    d.add_argument('--seed',type=int,default=92)
    d.add_argument('--out',type=Path,default=LOCAL/'demo')
    c=sub.add_parser('core',help='Formal core dataset 2-carrier paired run')
    c.add_argument('--datasets',nargs='+',choices=['madelon','hill_valley','musk1','arcene'],default=['madelon','hill_valley','musk1','arcene'])
    c.add_argument('--seeds',nargs='+',type=int,default=DEFAULT_SEEDS)
    c.add_argument('--out',type=Path,default=LOCAL/'table3/core')
    b=sub.add_parser('breadth',help='Formal five untouched breadth datasets, 2 carriers')
    b.add_argument('--datasets',nargs='+',choices=['ALLAML','colon','GLI-85','Prostate-GE','SMK-CAN-187'],default=['ALLAML','colon','GLI-85','Prostate-GE','SMK-CAN-187'])
    b.add_argument('--seeds',nargs='+',type=int,default=DEFAULT_SEEDS)
    b.add_argument('--out',type=Path,default=LOCAL/'table3/breadth')
    e=sub.add_parser('external',help='Core comparison MSF/HFMOEA/MBLPE, NSGA-II')
    e.add_argument('--datasets',nargs='+',choices=['madelon','hill_valley','musk1','arcene'],default=['madelon','hill_valley','musk1','arcene'])
    e.add_argument('--seeds',nargs='+',type=int,default=DEFAULT_SEEDS)
    e.add_argument('--methods',nargs='+',choices=['MSF-MOFS','HFMOEA-CP','MBLPE-CP'],default=['MSF-MOFS','HFMOEA-CP','MBLPE-CP'])
    e.add_argument('--out',type=Path,default=LOCAL/'table2_external')
    s=sub.add_parser('satellite',help='Single controlled synthetic telemetry subsystem; original runner')
    s.add_argument('--subsystem',choices=SUBSYSTEMS,required=True)
    s.add_argument('--start',type=int,default=192)
    s.add_argument('--end',type=int,default=201)
    s.add_argument('--out',type=Path,default=ROOT/'results/reproduced/satellite/table4' )
    sub.add_parser('satellite-all',help='Run four satellite subsystem experiments (slow) then aggregate')
    sub.add_parser('satellite-all-v3',help='Resume missing satellite seed rows without overwriting valid results')
    sp=sub.add_parser('sparse',help='Run portable 5-dataset Sparse-control search')
    sp.add_argument('--datasets',nargs='+',choices=['ALLAML','colon','GLI-85','Prostate-GE','SMK-CAN-187'],default=['ALLAML','colon','GLI-85','Prostate-GE','SMK-CAN-187'])
    sp.add_argument('--seeds',nargs='+',type=int,default=DEFAULT_SEEDS)
    sp.add_argument('--out',type=Path,default=LOCAL/'sparse_control')
    sub.add_parser('v3-status',help='Strict independent-search coverage and archive comparison')
    sub.add_parser('v4-status',help='Scope-aware complete v4 verification status')
    sub.add_parser('fisher-check',help='Recompute Fisher top-K Jaccard and Spearman from raw public datasets')
    sub.add_parser('replay-reliability',help='Reexecute 40 Fisher identity vs Sparse-control paired reliability searches')
    sub.add_parser('population50',help='Run original MADELON plus extension Hill-Valley/Prostate-GE (slow)')
    lp=sub.add_parser('population50-sample',help='Run specified population-50 seeds on all three datasets (incremental)')
    lp.add_argument('--seeds',nargs='+',type=int,default=[72])
    st=sub.add_parser('stats',help='Rebuild paired Bootstrap CIs and HV sensitivity STRICTLY from complete per-seed raw CSV(s)')
    st.add_argument('--scope',choices=['breadth','core','nine'],default='breadth')
    st.add_argument('--input',nargs='+',type=Path,default=None,help='Raw CSV(s); default uses archived five-breadth CSV for breadth, results/reference/ for core/nine')
    st.add_argument('--out',type=Path,default=None)
    st.add_argument('--evidence-type',choices=['archived_per_seed','reproduced','mixed'],default='archived_per_seed')
    sub.add_parser('reliability',help='Recalculate archive Fisher Jaccard and held-out association (not a new search)')
    sub.add_parser('status',help='Show completed end-to-end search cells and missing cells')
    vc=sub.add_parser('verify-core',help='Compare 40 rerun core MSF/NSGA-II rows with archived external-comparison seed records')
    vc.add_argument('--input',type=Path,default=OUT/'table3/core/raw.csv')
    vb=sub.add_parser('verify-breadth',help='Compare all 200 rerun breadth rows with historical raw per-seed results')
    vb.add_argument('--input',type=Path,default=OUT/'table3/breadth/raw.csv')
    mm=sub.add_parser('merge-results',aliases=['merge-core'],help='Combine independent rerun CSV shards; verify equal overlapping rows')
    mm.add_argument('--input',nargs='+',type=Path,required=True)
    mm.add_argument('--out',type=Path,default=LOCAL/'merged_core/raw.csv')
    a=p.parse_args(argv)
    if a.command=='check':
        out=LOCAL/'audit/report.json';report=run_archive_audit(ROOT,out)
        for name in ('data_hashes','musk_compat','table2','table3','public_table3_archived_reaggregation','table4','population50'):
            print(f'{name:16s} {report[name]["status"]}')
        print('ARCHIVED CHECKS PASS:',report['archived_checks_pass'])
        print('FULL END-TO-END REPRODUCED:',report['full_paper_end_to_end_reproduced'])
        print('Saved:',out)
        if not report['archived_checks_pass']: raise SystemExit(1)
    elif a.command=='smoke':
        _run(ROOT/'tests/smoke.py',[])
    elif a.command in ('demo','core'):
        args=(['--datasets',a.dataset,'--seeds',str(a.seed)] if a.command=='demo' else
              ['--datasets',*a.datasets,'--seeds',*map(str,a.seeds)])
        _run(ROOT/'scripts/public/run_v2_spea2_plugin.py',
             [*args,'--data-dir',str(CORE),'--out',str(a.out),'--pop',POP,'--gen',GENERATIONS])
        if a.command=='demo':
            comparison=compare_demo_with_archived_core(ROOT,a.out/'raw.csv',a.dataset,a.seed)
            (a.out/'single_seed_reference_check.json').write_text(json.dumps(comparison,indent=2)+'\n')
            print('Per-seed archived MSF-NSGA reference:',comparison['status'])
        else:
            # Available archived *MSF/NSGA-II* rows only; do not claim all four
            # experimental cells have a direct historical per-seed counterpart.
            import pandas as pd
            raw=pd.read_csv(a.out/'raw.csv')
            selected=raw[(raw.carrier=='NSGA-II')&(raw['init']=='MSF')]
            checks={f'{r.dataset}/{int(r.seed)}':compare_demo_with_archived_core(
                ROOT,a.out/'raw.csv',r.dataset,int(r.seed)) for r in selected.itertuples(index=False)}
            (a.out/'msf_nsga_archived_comparisons.json').write_text(
                json.dumps(checks,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
            nmatch=sum(x['status']=='MATCH' for x in checks.values())
            print(f'Available archived MSF-NSGA matches: {nmatch}/{len(checks)}; '
                  f'full core completion is separate; use `python -m msf_repro status`.')
    elif a.command=='breadth':
        _run(ROOT/'scripts/public/run_v2_untouched_breadth.py',
             ['--datasets',*a.datasets,'--seeds',*map(str,a.seeds),
              '--data-dir',str(BREADTH),'--out',str(a.out),'--pop',POP,'--gen',GENERATIONS])
    elif a.command=='sparse':
        _run(ROOT/'scripts/public/sparse_control_portable.py',
             ['--datasets',*a.datasets,'--seeds',*map(str,a.seeds),
              '--data-dir',str(BREADTH),'--out',str(a.out)],extra_paths=[ROOT/'scripts/public'])
    elif a.command=='v4-status':
        report=v4_status(ROOT)
        print(json.dumps(report,indent=2,ensure_ascii=False))
    elif a.command=='fisher-check':
        _run(ROOT/'src/msf_repro/fisher_recheck.py',['--root',str(ROOT),'--out',str(LOCAL/'fisher_stability')])
    elif a.command=='replay-reliability':
        _run(ROOT/'scripts/public/replay_reliability_portable.py',['--root',str(ROOT),'--out',str(LOCAL/'reliability_search')],extra_paths=[ROOT/'scripts/public'])
    elif a.command=='v3-status':
        report=verify_project(ROOT)
        for k,v in report.items():
            if k=='status_summary':continue
            print(f'{k:24s} {v["completed"]:3d}/{v["expected"]:3d} reference={v["reference_status"]}')
        print('Paper-wide complete:',report['status_summary']['full_paper_reproduced'])
        print('Report:', ROOT/'results/reproduced/paper/verification/v3/V3_EVIDENCE_STATUS.json')
    elif a.command=='external':
        _run(ROOT/'scripts/public/run_frozen_confirmation.py',
             ['--datasets',*a.datasets,'--seeds',*map(str,a.seeds),'--methods',*a.methods,
              '--data-dir',str(CORE),'--out',str(a.out),'--pop',POP,'--gen',GENERATIONS])
    elif a.command=='satellite':
        _satellite(a.subsystem,a.start,a.end,a.out)
        check=compare_satellite_with_archived(ROOT,a.out/a.subsystem/'per_seed_results.csv',a.subsystem)
        (a.out/a.subsystem/'per_seed_reference_check.json').write_text(json.dumps(check,indent=2)+'\n')
        print('Per-seed satellite archive comparison:',check['status'])
    elif a.command=='satellite-all':
        for name in SUBSYSTEMS:
            _satellite(name,192,201,ROOT/'results/reproduced/satellite/table4')
        _run(ROOT/'scripts/satellite/v4/analyze_msf_vs_random.py',
             ['--results-dir',str(ROOT/'results/reproduced/satellite/table4'),'--out',str(ROOT/'results/reproduced/satellite/table4/Table4.csv')])
    elif a.command=='satellite-all-v3':
        for name in SUBSYSTEMS:
            _satellite(name,192,201,ROOT/'results/reproduced/satellite/table4',resume=True)
        verify_project(ROOT)
    elif a.command=='stats':
        if a.input is None:
            if a.scope=='breadth':
                inputs=[ROOT/'evidence/public/untouched_breadth/raw.csv']
            elif a.scope=='core':
                inputs=[OUT/'table3/core/raw.csv']
            else:
                inputs=[OUT/'table3/core/raw.csv',OUT/'table3/breadth/raw.csv']
        else:
            inputs=a.input
        evidence=a.evidence_type if a.input is not None else (
            'archived_per_seed')
        dest=a.out or (LOCAL/'table3'/a.scope)
        meta=analyze_files(inputs,dest,scope=a.scope,evidence_type=evidence)
        print('Scope:',meta['scope'],'groups:',meta['groups'],'rows:',meta['raw_rows'])
        print('Provenance:',dest/'PROVENANCE.json')
        print('Paired CI:',dest/'paired_hv_ci_effect.csv')
        print('Reference sensitivity:',dest/'hv_reference_sensitivity_summary.csv')
    elif a.command in ('merge-results','merge-core'):
        result=merge_core_inputs(a.input,a.out)
        print('MERGED:',result['n_final_rows'],'unique cells; equal duplicate input rows:',result['deduplicated_rows'])
        print('Provenance:',a.out.parent/'merge_manifest.json')
    elif a.command=='verify-core':
        comparison=compare_core_msf_nsga_with_archive(ROOT,a.input)
        out=LOCAL/'verification/msf_nsga_40_archive_comparison.json';out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(comparison,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        print('Core MSF-NSGA 40 per-seed archived comparisons:',comparison['status'])
        print('Result:',out)
        if comparison['status']!='MATCH':raise SystemExit(1)
    elif a.command=='verify-breadth':
        comparison=compare_breadth_full_with_archive(ROOT,a.input)
        out=LOCAL/'verification/reference_per_seed_comparison.json';out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(comparison,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        print('Rerun vs archived per-seed breadth:',comparison['status'])
        print('Result:',out)
        if comparison['status']!='MATCH':raise SystemExit(1)
    elif a.command=='status':
        report=report_coverage(ROOT,LOCAL/'audit/run_coverage.json')
        for group in ('public_core','public_breadth'):
            meta=report[group]
            print(f'{group}: {meta["valid_rows"]}/{meta["expected_rows"]} paired-run cells')
            for ds,details in meta['by_dataset'].items():
                print(f'  {ds}: {details["completed"]}/{details["total"]}')
            if meta['extra_rows'] or meta['duplicate_rows']:
                print('WARNING nonstandard/duplicate rows:',meta['extra_rows'],meta['duplicate_rows'])
        print('ARCHIVED REFERENCE COVERAGE COMPLETE (NOT A NEW RERUN):',report['full_public_search_complete'])
        print('Saved:',LOCAL/'audit/run_coverage.json')
    elif a.command=='reliability':
        meta=reaggregate_reliability(ROOT,LOCAL/'reliability')
        print('ARCHIVE REAGGREGATION:',meta['status'])
        print('Dataset-centered held-out Spearman:',meta['dataset_centered_spearman']['delta_heldout'])
        print('Results:',LOCAL/'reliability/reliability_association_summary.csv')
        if meta['status']!='MATCH_ARCHIVED_REAGGREGATION': raise SystemExit(1)
    elif a.command=='population50-sample':
        _run(ROOT/'scripts/public/population50_madelon_portable.py',
             ['--core-dir',str(CORE),'--out',str(LOCAL/'population50/madelon'),
              '--seeds',*map(str,a.seeds)],extra_paths=[ROOT/'scripts/public'])
        _run(ROOT/'scripts/public/population50_extension.py',
             ['--core-dir',str(CORE),'--untouched-dir',str(BREADTH),
              '--out',str(LOCAL/'population50/extension'),'--seeds',*map(str,a.seeds)],
             extra_paths=[ROOT/'scripts/public'])
    elif a.command=='population50':
        _run(ROOT/'scripts/public/population50_madelon_portable.py',
             ['--core-dir',str(CORE),'--out',str(LOCAL/'population50/madelon')],
             extra_paths=[ROOT/'scripts/public'])
        _run(ROOT/'scripts/public/population50_extension.py',
             ['--core-dir',str(CORE),'--untouched-dir',str(BREADTH),
              '--out',str(LOCAL/'population50/extension')],extra_paths=[ROOT/'scripts/public'])


def _satellite(name:str, start:int,end:int,out:Path,resume:bool=False):
    args=['--data',str(ROOT/f'data/satellite/v8_confirmation/telemetry-msf-confirmation-{name}.csv'),
          '--out-dir',str(out/name),'--seed-start',str(start),'--seed-end',str(end)]
    if resume:args.append('--resume')
    _run(ROOT/'scripts/satellite/v4/run_msf_vs_random.py',args)


if __name__=='__main__':main()
