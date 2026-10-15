#!/usr/bin/env python3
"""Matched multi-seed MSF-MOFS vs Random experiment on satellite telemetry windows.

Primary endpoint mirrors the paper as closely as this single synthetic satellite
dataset permits: held-out Pareto-set hypervolume (HV), reference point (1, 1),
using objectives (1 - test G-mean, selected-feature ratio).

Seeds 92..101 are matched between MSF and Random. Both methods start from the
same Bernoulli(0.20) base population; MSF replaces exactly three chromosomes
with the Fisher top-k seeds.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("train_msf_mofs", HERE / "train_msf_mofs.py")
mod = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def nondominated_2d(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return minimization nondominated points, deduplicated and sorted by f1."""
    uniq = sorted(set((float(x), float(y)) for x, y in points))
    kept: list[tuple[float, float]] = []
    for i, p in enumerate(uniq):
        dominated = False
        for j, q in enumerate(uniq):
            if i == j:
                continue
            if q[0] <= p[0] and q[1] <= p[1] and (q[0] < p[0] or q[1] < p[1]):
                dominated = True
                break
        if not dominated:
            kept.append(p)
    return sorted(kept, key=lambda p: (p[0], -p[1]))


def hv2d_min(points: list[tuple[float, float]], ref=(1.0, 1.0)) -> float:
    front = nondominated_2d([(min(ref[0], max(0.0, x)), min(ref[1], max(0.0, y))) for x, y in points])
    if not front:
        return 0.0
    # For nondominated minimization front sorted by ascending x, y decreases.
    hv = 0.0
    prev_y = ref[1]
    for x, y in front:
        if y < prev_y:
            hv += (ref[0] - x) * (prev_y - y)
            prev_y = y
    return float(hv)


def heldout_front_metrics(pop, X_fit, y_fit, X_test, y_test, D: int):
    fronts = mod.rank_and_crowd(pop)
    training_front = fronts[0]
    records = []
    points = []
    for ind in training_front:
        cols = np.flatnonzero(ind.mask)
        if len(cols) == 0:
            g = 0.0
        else:
            g = mod.holdout_gmean(X_fit, y_fit, X_test, y_test, cols)
        rec = {
            "featureCount": int(len(cols)),
            "featureRatio": float(len(cols) / D),
            "searchCvGmean": float(1.0 - ind.obj[0]),
            "testGmean": float(g),
            "testError": float(1.0 - g),
        }
        records.append(rec)
        points.append((1.0 - g, len(cols) / D))
    hv = hv2d_min(points, (1.0, 1.0))
    heldout_nd = nondominated_2d(points)
    return hv, records, heldout_nd


def paired_rank_biserial(a: np.ndarray, b: np.ndarray, tol=1e-15) -> float:
    d = a - b
    nz = np.flatnonzero(np.abs(d) > tol)
    if len(nz) == 0:
        return 0.0
    absd = np.abs(d[nz])
    # average ranks for ties
    order = np.argsort(absd)
    ranks = np.empty(len(absd), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and math.isclose(absd[order[j]], absd[order[i]], rel_tol=0, abs_tol=tol):
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    wpos = float(ranks[d[nz] > 0].sum())
    wneg = float(ranks[d[nz] < 0].sum())
    denom = wpos + wneg
    return (wpos - wneg) / denom if denom else 0.0


def compare(a: Iterable[float], b: Iterable[float], higher_better=True) -> dict:
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    d = a - b
    wins = int(np.sum(d > 1e-12)) if higher_better else int(np.sum(d < -1e-12))
    losses = int(np.sum(d < -1e-12)) if higher_better else int(np.sum(d > 1e-12))
    ties = int(len(d) - wins - losses)
    if np.allclose(d, 0.0, atol=1e-15, rtol=0):
        p = 1.0
    else:
        p = float(wilcoxon(a, b, alternative="two-sided", zero_method="wilcox", method="auto").pvalue)
    rbc = paired_rank_biserial(a, b)
    if not higher_better:
        rbc = -rbc
    return {
        "aMean": float(np.mean(a)), "aStd": float(np.std(a, ddof=1)),
        "bMean": float(np.mean(b)), "bStd": float(np.std(b, ddof=1)),
        "meanDifference": float(np.mean((a-b) if higher_better else (b-a))),
        "wilcoxonP": p,
        "rankBiserialEffect": float(rbc),
        "winsTiesLosses": [wins, ties, losses],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "telemetry-msf-dataset.csv"))
    ap.add_argument("--seed-start", type=int, default=92)
    ap.add_argument("--seed-end", type=int, default=101)
    ap.add_argument("--search-samples", type=int, default=2400)
    ap.add_argument("--out-dir", default=str(HERE / "experiments" / "multiseed_92_101"))
    args = ap.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    feature_names = [c for c in df.columns if c not in mod.META]
    df = df[df.label.isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)
    train_df = df[df.split == "train"].copy()
    val_df = df[df.split == "val"].copy()
    test_df = df[df.split == "test"].copy()
    medians = train_df[feature_names].median(axis=0).fillna(0.0)
    for part in (train_df, val_df, test_df):
        part.loc[:, feature_names] = part[feature_names].fillna(medians)

    X_train = train_df[feature_names].to_numpy(float); y_train = train_df.label.to_numpy(int)
    X_val = val_df[feature_names].to_numpy(float); y_val = val_df.label.to_numpy(int)
    X_test = test_df[feature_names].to_numpy(float); y_test = test_df.label.to_numpy(int)
    X_fit = np.vstack([X_train, X_val]); y_fit = np.concatenate([y_train, y_val])
    D = len(feature_names)
    fisher = mod.fisher_scores(X_train, y_train)
    fisher_order = np.argsort(-fisher)

    rows = []
    detail = {
        "schemaVersion": "msf-satellite-multiseed-1.0",
        "seeds": list(range(args.seed_start, args.seed_end + 1)),
        "dataset": {
            "candidateFeatures": D,
            "trainRows": len(train_df), "valRows": len(val_df), "testRows": len(test_df),
            "guidedCardinalities": mod.msf_cardinalities(D),
        },
        "runs": [],
    }

    all_full_g = mod.holdout_gmean(X_fit, y_fit, X_test, y_test, np.arange(D))

    for seed in range(args.seed_start, args.seed_end + 1):
        search_X, search_y = mod.balanced_subsample(X_train, y_train, args.search_samples, seed)
        run_detail = {"seed": seed, "methods": {}}
        run_row = {"seed": seed}
        for name, guided in (("MSF", True), ("Random", False)):
            pop, cache = mod.evolve(search_X, search_y, fisher_order, seed=seed, guided=guided)
            chosen, pareto_val = mod.choose_from_pareto(pop, X_train, y_train, X_val, y_val)
            idx = np.flatnonzero(chosen.mask)
            chosen_test_g = mod.holdout_gmean(X_fit, y_fit, X_test, y_test, idx)
            hv, heldout_records, heldout_nd = heldout_front_metrics(pop, X_fit, y_fit, X_test, y_test, D)
            method = {
                "heldoutParetoHV": hv,
                "chosenFeatureCount": int(len(idx)),
                "chosenFeatureRatio": float(len(idx)/D),
                "chosenTestGmean": float(chosen_test_g),
                "chosenFeatures": [feature_names[i] for i in idx],
                "uniqueWrapperEvaluations": int(len(cache)),
                "paretoValidation": pareto_val,
                "heldoutTrainingFront": heldout_records,
                "heldoutNondominatedObjectives": [[float(x), float(y)] for x, y in heldout_nd],
            }
            run_detail["methods"][name] = method
            prefix = name.lower()
            run_row[f"{prefix}_hv"] = hv
            run_row[f"{prefix}_feature_count"] = int(len(idx))
            run_row[f"{prefix}_test_gmean"] = float(chosen_test_g)
            run_row[f"{prefix}_unique_evals"] = int(len(cache))
        rows.append(run_row)
        detail["runs"].append(run_detail)
        print(f"seed={seed} MSF HV={run_row['msf_hv']:.6f} Random HV={run_row['random_hv']:.6f} | "
              f"G={run_row['msf_test_gmean']:.4f}/{run_row['random_test_gmean']:.4f} | "
              f"K={run_row['msf_feature_count']}/{run_row['random_feature_count']}")

    rdf = pd.DataFrame(rows)
    summary = {
        "primaryEndpoint": "held-out Pareto-set HV with reference point (1,1)",
        "allFeatures5nnTestGmean": float(all_full_g),
        "heldoutParetoHV": compare(rdf.msf_hv, rdf.random_hv, higher_better=True),
        "chosenTestGmean": compare(rdf.msf_test_gmean, rdf.random_test_gmean, higher_better=True),
        "chosenFeatureCount": compare(rdf.msf_feature_count, rdf.random_feature_count, higher_better=False),
        "uniqueWrapperEvaluations": compare(rdf.msf_unique_evals, rdf.random_unique_evals, higher_better=False),
    }
    detail["summary"] = summary

    rdf.to_csv(outdir / "per_seed_results.csv", index=False)
    (outdir / "experiment_results.json").write_text(json.dumps(detail, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = []
    md.append("# MSF-MOFS vs Random：卫星遥测多随机种子实验\n")
    md.append(f"- Seeds: {args.seed_start}–{args.seed_end}\n- D={D}; guided seeds={mod.msf_cardinalities(D)}; population=12; generations=6\n")
    md.append("- Primary endpoint: held-out Pareto-set HV, reference point (1,1)\n")
    md.append("- Train/val/test split is scenario-level; transition samples are excluded from MSF training/evaluation.\n")
    md.append(f"- All-288-feature 5-NN test G-mean: **{all_full_g:.4f}**\n")
    md.append("\n## Per-seed results\n")
    md.append("|Seed|MSF HV|Random HV|MSF G-mean|Random G-mean|MSF K|Random K|\n|---:|---:|---:|---:|---:|---:|---:|\n")
    for _, r in rdf.iterrows():
        md.append(f"|{int(r.seed)}|{r.msf_hv:.6f}|{r.random_hv:.6f}|{r.msf_test_gmean:.4f}|{r.random_test_gmean:.4f}|{int(r.msf_feature_count)}|{int(r.random_feature_count)}|\n")
    md.append("\n## Paired summary\n")
    for label, key in [("Held-out Pareto HV", "heldoutParetoHV"), ("Chosen test G-mean", "chosenTestGmean"), ("Chosen feature count (lower better)", "chosenFeatureCount"), ("Unique wrapper evaluations (lower better)", "uniqueWrapperEvaluations")]:
        s = summary[key]
        md.append(f"- **{label}**: MSF={s['aMean']:.6f}±{s['aStd']:.6f}, Random={s['bMean']:.6f}±{s['bStd']:.6f}; "
                  f"W/T/L={s['winsTiesLosses'][0]}/{s['winsTiesLosses'][1]}/{s['winsTiesLosses'][2]}; "
                  f"Wilcoxon p={s['wilcoxonP']:.6g}; rank-biserial={s['rankBiserialEffect']:.3f}.\n")
    md.append("\n## Interpretation boundary\n")
    md.append("This is a fixed synthetic satellite dataset with repeated search seeds, not multiple independent real satellites/datasets. Statistical significance here quantifies search-seed robustness on this dataset; it must not be presented as evidence of real-orbit clinical/engineering generalization.\n")
    (outdir / "REPORT.md").write_text("".join(md), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Outputs: {outdir}")

if __name__ == "__main__":
    main()
