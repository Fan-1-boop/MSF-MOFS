from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.model_selection import train_test_split

from irmofs.highdim_datasets import load_highdim_dataset
from irmofs.environments import make_nested_ir_environments
from irmofs.evaluator import FeatureSubsetEvaluator, heldout_gmeans
from irmofs.stable_guidance import fisher_scores, make_cisr_multiscale_population
from irmofs.highdim_formal import make_base_population
from irmofs.nsga2 import run_nsga2, hypervolume_2d
from irmofs.external_baselines import (
    _hfmoea_filter_rankings,
    run_hfmoea_adapted,
    run_mblpe_adapted,
)

METHODS = ["MSF-MOFS", "HFMOEA-CP", "MBLPE-CP"]


def front_test_metrics(pareto_pop, train_envs, X_test, y_test, k=5):
    objs, means, worsts, counts = [], [], [], []
    for chromosome in pareto_pop:
        tg = heldout_gmeans(chromosome, train_envs, X_test, y_test, k)
        vals = np.asarray([tg[ir] for ir in sorted(tg)], dtype=float)
        mean_g = float(vals.mean())
        means.append(mean_g)
        worsts.append(float(vals.min()))
        counts.append(int(np.sum(chromosome)))
        objs.append([1.0 - mean_g, float(np.mean(chromosome))])
    F = np.asarray(objs, dtype=float)
    return {
        "test_pf_hv": float(hypervolume_2d(F)),
        "pf_size": int(len(pareto_pop)),
        "best_test_mean_gmean_on_pf_diagnostic": float(np.max(means)),
        "best_test_worst_gmean_on_pf_diagnostic": float(np.max(worsts)),
        "min_features_on_pf": int(np.min(counts)),
    }


def rank_biserial_paired(diff: np.ndarray) -> float:
    d = np.asarray(diff, float)
    nz = d[~np.isclose(d, 0.0, atol=1e-15, rtol=0.0)]
    if len(nz) == 0:
        return 0.0
    r = rankdata(np.abs(nz), method="average")
    wp = float(r[nz > 0].sum())
    wm = float(r[nz < 0].sum())
    return 0.0 if wp + wm == 0 else (wp - wm) / (wp + wm)


def holm_adjust(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    out = np.empty(len(p), float)
    running = 0.0
    m = len(p)
    for j, idx in enumerate(order):
        running = max(running, (m - j) * p[idx])
        out[idx] = min(1.0, running)
    return out


def write_derived(raw_path: Path, out_dir: Path):
    df = pd.read_csv(raw_path)
    if df.empty:
        return
    summary = df.groupby(["dataset", "method"], as_index=False).agg(
        n=("seed", "count"),
        train_hv_mean=("train_hv", "mean"),
        train_hv_std=("train_hv", "std"),
        test_pf_hv_mean=("test_pf_hv", "mean"),
        test_pf_hv_std=("test_pf_hv", "std"),
        unique_wrapper_evaluations_mean=("unique_wrapper_evaluations", "mean"),
        unique_wrapper_evaluations_std=("unique_wrapper_evaluations", "std"),
        ranking_or_filter_time_mean=("ranking_or_filter_time", "mean"),
        ranking_or_filter_time_std=("ranking_or_filter_time", "std"),
        search_time_mean=("search_time", "mean"),
        search_time_std=("search_time", "std"),
        heldout_evaluation_time_mean=("heldout_evaluation_time", "mean"),
        heldout_evaluation_time_std=("heldout_evaluation_time", "std"),
        total_time_mean=("total_time", "mean"),
        total_time_std=("total_time", "std"),
    )
    summary.to_csv(out_dir / "confirmation_summary.csv", index=False)

    # Predeclared primary confirmatory tests: held-out PF-HV, MSF vs each external baseline,
    # Holm correction across those two comparisons within each dataset.
    test_rows = []
    for ds, g in df.groupby("dataset"):
        local = []
        pvals = []
        for base in ["HFMOEA-CP", "MBLPE-CP"]:
            a = g[g.method == "MSF-MOFS"].set_index("seed")["test_pf_hv"]
            b = g[g.method == base].set_index("seed")["test_pf_hv"]
            idx = a.index.intersection(b.index)
            if len(idx) == 0:
                continue
            diff = (a.loc[idx] - b.loc[idx]).to_numpy(float)
            try:
                stat, p = wilcoxon(diff, zero_method="wilcox", alternative="two-sided", method="auto")
                stat, p = float(stat), float(p)
            except ValueError:
                stat, p = 0.0, 1.0
            tol = 1e-15
            rec = {
                "dataset": ds,
                "candidate": "MSF-MOFS",
                "baseline": base,
                "metric": "test_pf_hv",
                "n": int(len(diff)),
                "mean_difference": float(diff.mean()),
                "median_difference": float(np.median(diff)),
                "wins": int((diff > tol).sum()),
                "ties": int((np.abs(diff) <= tol).sum()),
                "losses": int((diff < -tol).sum()),
                "rank_biserial": float(rank_biserial_paired(diff)),
                "wilcoxon_stat": stat,
                "p_raw": p,
            }
            local.append(rec); pvals.append(p)
        if local:
            adj = holm_adjust(pvals)
            for rec, q in zip(local, adj):
                rec["p_holm_within_dataset"] = float(q)
                test_rows.append(rec)
    pd.DataFrame(test_rows).to_csv(out_dir / "confirmation_paired_tests.csv", index=False)

    # Dataset-level average ranks from mean held-out PF-HV. Only complete dataset triplets enter.
    rank_rows = []
    for ds, g in summary.groupby("dataset"):
        if set(g.method) != set(METHODS):
            continue
        means = g.set_index("method")["test_pf_hv_mean"].reindex(METHODS)
        ranks = rankdata(-means.to_numpy(float), method="average")
        for m, r in zip(METHODS, ranks):
            rank_rows.append({"dataset": ds, "method": m, "rank": float(r)})
    if rank_rows:
        rd = pd.DataFrame(rank_rows)
        avg = rd.groupby("method", as_index=False)["rank"].mean().rename(columns={"rank":"average_rank"})
        rd.to_csv(out_dir / "confirmation_dataset_ranks.csv", index=False)
        avg.sort_values("average_rank").to_csv(out_dir / "confirmation_average_ranks.csv", index=False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["madelon", "hill_valley", "musk1", "arcene"])
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(72,82)))
    p.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pop", type=int, default=12)
    p.add_argument("--gen", type=int, default=6)
    return p.parse_args()


def main():
    a = parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    raw_path = a.out / "confirmation_raw.csv"
    if raw_path.exists():
        rows = pd.read_csv(raw_path).to_dict("records")
    else:
        rows = []
    done = {(str(r["dataset"]), int(r["seed"]), str(r["method"])) for r in rows}

    protocol = {
        "datasets": a.datasets,
        "seeds": a.seeds,
        "methods": a.methods,
        "population": a.pop,
        "generations": a.gen,
        "outer_split": "stratified 70/30, random_state=seed",
        "wrapper": "5-NN, 3-fold, mean G-mean across controlled IR={1,3,5}, minimize feature ratio",
        "msf": "Full outer-training Fisher; matched random base population; replace exactly 3 chromosomes with smallest 3 distinct sizes from {10,0.05D,0.10D,0.20D}; standard NSGA-II thereafter",
        "timing_scope": "method-specific only; excludes dataset loading, outer split, and shared IR-environment construction; includes ranking/filter preprocessing, search, and held-out PF evaluation",
        "frozen_confirmation": True,
    }
    (a.out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    for ds in a.datasets:
        bundle = load_highdim_dataset(ds, a.data_dir)
        D = int(bundle.X.shape[1])
        for seed in a.seeds:
            Xtr, Xte, ytr, yte = train_test_split(
                bundle.X, bundle.y, test_size=0.30, stratify=bundle.y, random_state=seed
            )
            envs = make_nested_ir_environments(Xtr, ytr, [1,3,5], 3, seed)

            for method in a.methods:
                key = (ds, int(seed), method)
                if key in done:
                    continue
                evaluator = FeatureSubsetEvaluator(envs, mode="mean", n_neighbors=5, backend="sklearn")
                method_t0 = time.perf_counter()
                seed_sizes = ""

                # PREPROCESSING / GUIDANCE ONLY
                prep_t0 = time.perf_counter()
                if method == "MSF-MOFS":
                    scores = fisher_scores(Xtr, ytr)
                    base = make_base_population(a.pop, D, seed, 0.20)
                    # Freeze exactly 3 guided chromosomes independent of population size.
                    # For P0-1 pop=12, guided_fraction=.25 gives exactly 3.
                    init, sizes = make_cisr_multiscale_population(base, scores, guided_fraction=3.0/a.pop)
                    if len(sizes) != 3:
                        raise RuntimeError(f"MSF must inject exactly 3 guided chromosomes, got {sizes}")
                    seed_sizes = "/".join(map(str, sizes))
                elif method == "HFMOEA-CP":
                    rankings = _hfmoea_filter_rankings(Xtr, ytr, seed)
                    seed_sizes = "10-filter top10"
                else:
                    T = min(D, a.pop * 3)
                    m_scores = fisher_scores(Xtr, ytr)
                    selected = np.argsort(-m_scores, kind="stable")[:T]
                    seed_sizes = f"Fisher prefilter T={T}"
                prep_time = time.perf_counter() - prep_t0

                # SEARCH (training wrapper evaluations)
                search_t0 = time.perf_counter()
                if method == "MSF-MOFS":
                    result = run_nsga2(
                        evaluator.evaluate, D, population_size=a.pop, generations=a.gen,
                        random_state=seed, verbose=False, initial_population=init,
                    )
                elif method == "HFMOEA-CP":
                    result = run_hfmoea_adapted(
                        evaluator.evaluate, Xtr, ytr, D, a.pop, a.gen, seed,
                        precomputed_rankings=rankings,
                    )
                else:
                    result = run_mblpe_adapted(
                        evaluator.evaluate, Xtr, ytr, D, a.pop, a.gen, seed,
                        prefilter_selected=selected,
                    )
                search_time = time.perf_counter() - search_t0
                train_hv = float(hypervolume_2d(result.pareto_objectives))
                unique_evals = int(len(evaluator.cache))

                # HELD-OUT PARETO-SET EVALUATION ONLY
                held_t0 = time.perf_counter()
                fm = front_test_metrics(result.pareto_population, envs, Xte, yte, 5)
                held_time = time.perf_counter() - held_t0
                total = time.perf_counter() - method_t0

                row = {
                    "dataset": ds,
                    "seed": int(seed),
                    "method": method,
                    "n_samples": int(bundle.X.shape[0]),
                    "n_features": D,
                    "train_hv": train_hv,
                    **fm,
                    "unique_wrapper_evaluations": unique_evals,
                    "seed_sizes": seed_sizes,
                    "ranking_or_filter_time": float(prep_time),
                    "search_time": float(search_time),
                    "heldout_evaluation_time": float(held_time),
                    "total_time": float(total),
                }
                rows.append(row); done.add(key)
                pd.DataFrame(rows).to_csv(raw_path, index=False)
                write_derived(raw_path, a.out)
                print(json.dumps(row, ensure_ascii=False), flush=True)

    write_derived(raw_path, a.out)

if __name__ == "__main__":
    main()
