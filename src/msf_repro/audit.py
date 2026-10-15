"""Read-only audit of ARCHIVED results. No archived evidence is overwritten.

Important: recreating a table from archived per-seed rows is NOT a full algorithm rerun.
"""
from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SUBSYSTEMS = ("power", "thermal", "attitude", "data")
PAPER_TABLE2 = {
    "madelon": [0.7555, 0.7554, 0.6297],
    "hill_valley": [0.3616, 0.3205, 0.3464],
    "musk1": [0.7075, 0.6996, 0.6902],
    "arcene": [0.6728, 0.6057, 0.5665],
}
PAPER_TABLE3_NS = {
    "madelon": 0.408, "hill_valley": 0.026, "musk1": 0.029,
    "arcene": 0.163, "ALLAML": 0.504, "colon": 0.388,
    "GLI-85": 0.396, "Prostate-GE": 0.394, "SMK-CAN-187": 0.162,
}
PAPER_TABLE3_SPEA2 = {
    "madelon": 0.406, "hill_valley": 0.035, "musk1": 0.027,
    "arcene": 0.122, "ALLAML": 0.522, "colon": 0.416,
    "GLI-85": 0.381, "Prostate-GE": 0.397, "SMK-CAN-187": 0.164,
}
PAPER_TABLE4 = {"power": 0.0832, "thermal": 0.0464, "attitude": 0.0380, "data": 0.1266}


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required archived evidence missing: {path}")
    return pd.read_csv(path)


def _holm(ps):
    p = np.array(ps, dtype=float)
    order = np.argsort(p, kind="stable")
    result = np.empty_like(p)
    running = 0.
    for j, i in enumerate(order):
        running = max(running, float((len(p)-j)*p[i]))
        result[i] = min(1.,running)
    return result


def _check_hashes(root: Path) -> dict:
    manifest = root / "data/public/DATASET_MANIFEST.csv"
    bad, missing, n = [], [], 0
    with manifest.open(encoding="utf-8", newline="") as file:
        for rec in csv.DictReader(file):
            folder = "core" if rec["group"] == "core_or_development_archive" else "breadth"
            path = root / "data/public" / folder / rec["filename"]
            if not path.is_file():
                missing.append(rec["filename"])
                continue
            sha = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
            n += 1
            if sha != rec["sha256"]:
                bad.append(rec["filename"])
    return dict(status="PASS" if not bad and not missing and n == 11 else "FAIL",
                verified=n, expected=11, mismatches=bad, missing=missing)


def _musk_compat(root: Path) -> dict:
    meta_path = root / 'data/public/core/MUSK1_DECOMPRESSION.json'
    meta=json.loads(meta_path.read_text(encoding='utf-8'))
    archive=root/'data/public/core'/meta['source_archive']
    output=root/'data/public/core'/meta['decoded_file']
    errors=[]
    if not output.is_file():
        errors.append('Derived portable Musk1 file missing')
    else:
        if hashlib.sha256(output.read_bytes()).hexdigest()!=meta['decoded_file_sha256']:
            errors.append('Decoded Musk1 SHA256 mismatch')
        if output.stat().st_size != meta['decoded_file_bytes']:
            errors.append('Decoded Musk1 byte count mismatch')
    if not archive.is_file():
        errors.append('Original Musk1 ZIP missing')
    else:
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=meta['source_archive_sha256']:
            errors.append('Original Musk1 ZIP SHA256 mismatch')
        with zipfile.ZipFile(archive) as z:
            member=z.read(meta['member'])
        if hashlib.sha256(member).hexdigest()!=meta['compressed_member_sha256']:
            errors.append('Compressed Musk1 member SHA256 mismatch')
    return dict(status='PASS' if not errors else 'FAIL', errors=errors,
                scope='source archive, compressed member and portable decoded bytes SHA-256')


def _table2(root: Path) -> dict:
    raw = _read(root / "evidence/public/core_external_confirmation/confirmation_raw.csv")
    methods = ("MSF-MOFS", "HFMOEA-CP", "MBLPE-CP")
    values = {}
    bad = []
    for ds, expected in PAPER_TABLE2.items():
        for method, want in zip(methods, expected):
            recorded_dataset = "musk" if ds == "musk1" else ds
            g = raw[(raw.dataset == recorded_dataset) & (raw.method == method)]
            got = float(g.test_pf_hv.mean())
            values[f"{ds}/{method}"] = dict(n=len(g), mean=got, std=float(g.test_pf_hv.std(ddof=1)))
            if len(g) != 10 or abs(got-want) > 0.000055:
                bad.append(f"{ds}/{method}: got {got:.6f}, expected rounded {want:.4f}, n={len(g)}")
    return dict(status="PASS" if not bad else "FAIL", level="archived_per_seed_reaggregation",
                checked=len(values), errors=bad, values=values)


def _table3(root: Path) -> dict:
    # The ZIP lacks the full 4-core x 2-carrier x 2-init raw input to the camera-ready CI script.
    archived = _read(root / "evidence/revision/statistics_existing_results/paired_hv_ci_effect_9dataset.csv")
    breadth = _read(root / "evidence/public/untouched_breadth/raw.csv")
    errors, raw_checked = [], 0
    for (ds, carrier), g in breadth.groupby(["dataset", "carrier"]):
        piv = g.pivot(index="seed", columns="init", values="test_pf_hv").dropna()
        if len(piv) != 10 or not {'MSF','Random'}.issubset(piv.columns):
            errors.append(f"{ds}/{carrier}: invalid raw grouping")
            continue
        diff = piv.MSF-piv.Random
        ar = archived[(archived.dataset == ds) & (archived.carrier == carrier)]
        if len(ar) != 1 or abs(float(ar.mean_delta.iloc[0])-float(diff.mean())) > 1e-10:
            errors.append(f"{ds}/{carrier}: archived delta differs from source raw")
        raw_checked += 1
    for carrier, want in (("NSGA-II", PAPER_TABLE3_NS), ("SPEA2", PAPER_TABLE3_SPEA2)):
        subset = archived[archived.carrier == carrier]
        if len(subset) != 9 or int(subset.wins.sum()) != 87 or int(subset.losses.sum()) != 3:
            errors.append(f"{carrier}: unexpected nine-dataset win counts")
        if int((subset.p_holm_within_carrier < .05).sum()) != 8:
            errors.append(f"{carrier}: unexpected Holm significance count")
        for ds, exp in want.items():
            row = subset[subset.dataset == ds]
            if len(row) != 1 or abs(float(row.mean_delta.iloc[0])-exp) > .0006:
                errors.append(f"{carrier}/{ds}: delta not aligned with rounded paper value")
    return dict(status="PASS_ARCHIVE_ONLY" if not errors else "FAIL",
                level="10 breadth groups independently aggregated; 8 core groups checked against archived summary only",
                archived_dataset_carrier_groups=len(archived),
                independently_recomputed_breadth_groups=raw_checked,
                core_raw_missing=True,
                errors=errors)


def _table4(root: Path) -> dict:
    errors, rows, ps = [], [], []
    for name in SUBSYSTEMS:
        x = _read(root / f"evidence/satellite/raw_per_seed/{name}/per_seed_results.csv")
        diff = x.msf_hv - x.random_hv
        p = float(wilcoxon(x.msf_hv, x.random_hv, alternative="two-sided").pvalue)
        if len(x) != 10 or abs(float(diff.mean())-PAPER_TABLE4[name]) > 0.000055:
            errors.append(f"{name}: unexpected mean effect or sample count")
        if int((diff > 0).sum()) != 10:
            errors.append(f"{name}: not 10 positive paired runs")
        rows.append(dict(subsystem=name,n=len(x),mean_msf=float(x.msf_hv.mean()),
                         mean_random=float(x.random_hv.mean()),delta=float(diff.mean()),
                         msf_sd=float(x.msf_hv.std(ddof=1)),random_sd=float(x.random_hv.std(ddof=1)),p_raw=p))
        ps.append(p)
    for row, q in zip(rows,_holm(ps)):
        row['p_holm'] = float(q)
        if abs(q - 0.0078125)>1e-12: errors.append(f"{row['subsystem']} unexpected Holm p={q}")
    return dict(status="PASS" if not errors else "FAIL", level="archived_per_seed_reaggregation",
                rows=rows, errors=errors)


def _population50(root: Path) -> dict:
    a = _read(root / "evidence/public/P0_72_81_AND_MECHANISM/P0_4/large_population_raw.csv")
    b = _read(root / "evidence/revision/population50_extension/large_population_extension_raw.csv")
    both = pd.concat([a,b], ignore_index=True)
    errors, means = [], {}
    paper = {'madelon': .4165, 'hill_valley': .0312, 'Prostate-GE': .4217}
    for ds, exp in paper.items():
        sub=both[both.dataset.str.lower()==ds.lower()]
        piv=sub.pivot_table(index='seed', columns='method', values='gen1_test_pf_hv').dropna()
        if not {'MSF-MOFS','Random'}.issubset(piv.columns):
            errors.append(f'{ds}: missing init methods');continue
        delta=piv['MSF-MOFS']-piv['Random']
        means[ds] = dict(n=len(delta),mean_delta=float(delta.mean()),wins=int((delta>0).sum()))
        if len(delta)!=10 or abs(delta.mean()-exp)>.000055 or (delta>0).sum()!=10:
            errors.append(f'{ds}: large-population archived results mismatch')
        if not (sub[sub.method == 'MSF-MOFS'].guided_chromosomes == 3).all():
            errors.append(f'{ds}: population=50 guided count !=3')
    return dict(status='PASS' if not errors else 'FAIL',level='archived_per_seed_reaggregation',means=means,errors=errors)


def run_archive_audit(root: Path, output: Path | None = None) -> dict:
    root = root.resolve()
    report={
        'scope':'Archived evidence and static integrity. NOT full end-to-end paper reproduction.',
        'data_hashes':_check_hashes(root),
        'musk_compat':_musk_compat(root),
        'table2':_table2(root),
        'table3':_table3(root),
        'table4':_table4(root),
        'population50':_population50(root),
        'unresolved':[
            'Nine-dataset original 4-core-carrier paired MSF/Random raw CSV absent; Table 3 complete bootstrap CI not independently regenerated.',
            'Six-dataset development schedule full per-seed table not bundled.',
            'Some satellite V8 telemetry-generation JavaScript requires missing historical modules.',
            'Frozen 140-feature extraction list/model and independent frame-timing execution not bundled.',
            'No real-orbit telemetry validation. Controlled synthetic scenario only.',
            'SPEA2 source is validated reconstructed implementation, not byte-for-byte original.'
        ],
    }
    # V2: separate NEW algorithm-rerun evidence from stored historical summaries.
    # The table3 field above intentionally retains its legacy archive-only meaning.
    from .coverage import report_coverage
    coverage=report_coverage(root)
    if coverage['full_public_search_complete']:
        try:
            from .statistics import analyze_files
            output_dir=root/'results/reproduced/paper/audit/public_table3_recomputed'
            result=analyze_files([root/'results/reference/table3/core/raw.csv',
                                  root/'results/reference/table3/breadth/raw.csv'],output_dir,
                                 scope='nine',evidence_type='archived_per_seed')
            status=result.get('archive_summary_comparison','MISSING_REFERENCE')
            report['public_table3_archived_reaggregation']={
                'status':'PASS_ARCHIVED_360_AGGREGATION' if status=='MATCH' else 'FAIL',
                'n_archived_search_rows':result['raw_rows'],
                'n_groups':result['groups'],
                'archive_summary_comparison':status,
                'full_paper_scope':False}
        except Exception as exc:
            report['public_table3_archived_reaggregation']={'status':'FAIL','error':str(exc)}
    else:
        report['public_table3_archived_reaggregation']={'status':'INCOMPLETE',
            'public_core_new_rows':coverage['public_core']['valid_rows'],
            'public_breadth_new_rows':coverage['public_breadth']['valid_rows']}
    report['archived_checks_pass'] = all(report[x]['status'] in ('PASS','PASS_ARCHIVE_ONLY')
                                         for x in ('data_hashes','musk_compat','table2','table3','table4','population50'))
    report['full_paper_end_to_end_reproduced'] = False
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return report
