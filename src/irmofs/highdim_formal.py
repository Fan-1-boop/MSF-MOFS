# -*- coding: utf-8 -*-
# MSF-MOFS 正式复现源码。
# 中文阅读说明：本文件保留原版所有可执行语句与随机数调用顺序；本次仅增加中文注释。
# 修改优化器、目标函数或评估协议之前，必须重新进行完整论文结果回归。
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split

from .highdim_datasets import load_highdim_dataset
from .environments import make_nested_ir_environments
from .evaluator import FeatureSubsetEvaluator, heldout_gmeans
from .stable_guidance import cross_ir_stable_relevance
from .nsga2 import run_nsga2, hypervolume_2d
from .selection import select_ideal_distance


def make_base_population(n: int, d: int, seed: int, feature_prob: float = 0.2) -> np.ndarray:
    rng = np.random.default_rng(seed + 170_003)
    pop = (rng.random((n, d)) < feature_prob).astype(np.uint8)
    for i in range(n):
        if pop[i].sum() == 0:
            pop[i, rng.integers(0, d)] = 1
    return pop


def matched_guidance(
    base: np.ndarray,
    scores: np.ndarray,
    seed: int,
    guided_fraction: float = 0.25,
    expected_feature_ratio: float = 0.2,
) -> np.ndarray:
    """Replace only the first guided_fraction of a matched base population.

    All methods share the remaining base individuals, and run_nsga2 receives an
    explicit initial population for every method so its evolutionary RNG stream is
    not shifted by internal initialization.
    """
    pop = np.asarray(base, dtype=np.uint8).copy()
    s = np.asarray(scores, dtype=float)
    n, d = pop.shape
    n_guided = int(round(n * guided_fraction))
    if n_guided == 0:
        return pop

    order = np.argsort(-s, kind="stable")
    ks = sorted(set(max(1, int(round(d * r))) for r in (0.05, 0.10, 0.20)))
    row = 0
    for k in ks:
        if row >= n_guided:
            break
        pop[row] = 0
        pop[row, order[:k]] = 1
        row += 1

    if row < n_guided:
        rng = np.random.default_rng(seed + 271_828)
        w = 0.25 + 1.5 * s
        probs = expected_feature_ratio * w / max(float(w.mean()), 1e-12)
        probs = np.clip(probs, 0.01, 0.75)
        while row < n_guided:
            bits = (rng.random(d) < probs).astype(np.uint8)
            if bits.sum() == 0:
                bits[int(np.argmax(s))] = 1
            pop[row] = bits
            row += 1
    return pop


def test_front_metrics(pareto_pop, train_envs, X_test, y_test, k=5):
    objs = []
    means = []
    worsts = []
    for chromosome in pareto_pop:
        tg = heldout_gmeans(chromosome, train_envs, X_test, y_test, k)
        vals = np.asarray([tg[ir] for ir in sorted(tg)], dtype=float)
        mean_g = float(vals.mean())
        means.append(mean_g)
        worsts.append(float(vals.min()))
        objs.append([1.0 - mean_g, float(np.mean(chromosome))])
    F = np.asarray(objs, dtype=float)
    return {
        "test_pf_hv": hypervolume_2d(F),
        "test_pf_best_mean_gmean": float(np.max(means)),
        "test_pf_best_worst_gmean": float(np.max(worsts)),
        "test_pf_size": int(len(pareto_pop)),
    }


def holm_adjust(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * p[idx]
        running = max(running, value)
        adj[idx] = min(1.0, running)
    return adj.tolist()


def paired_tests(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "train_hv",
        "test_pf_hv",
        "test_mean_gmean",
        "test_worst_gmean",
        "selected_features",
    ]
    comps = [("stable25", "mean"), ("stable25", "permuted_control")]
    rows = []
    for ds, g in runs.groupby("dataset"):
        for cand, base in comps:
            p_list = []
            local = []
            for metric in metrics:
                a = g[g.method == cand].set_index("seed")[metric]
                b = g[g.method == base].set_index("seed")[metric]
                idx = a.index.intersection(b.index)
                diff = (a.loc[idx] - b.loc[idx]).astype(float)
                try:
                    stat, p = wilcoxon(diff.values, zero_method="wilcox", alternative="two-sided")
                    p = float(p)
                except ValueError:
                    stat, p = 0.0, 1.0
                rec = {
                    "dataset": ds,
                    "candidate": cand,
                    "baseline": base,
                    "metric": metric,
                    "n": int(len(diff)),
                    "mean_difference": float(diff.mean()),
                    "median_difference": float(diff.median()),
                    "wins": int((diff > 0).sum()),
                    "ties": int((diff == 0).sum()),
                    "losses": int((diff < 0).sum()),
                    "wilcoxon_stat": float(stat),
                    "p_raw": p,
                }
                local.append(rec)
                p_list.append(p)
            adj = holm_adjust(p_list)
            for rec, q in zip(local, adj):
                rec["p_holm_within_comparison"] = q
                rows.append(rec)
    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["madelon", "hill_valley", "musk1"])
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--pop", type=int, default=12)
    p.add_argument("--gen", type=int, default=6)
    p.add_argument("--guided-fraction", type=float, default=0.25)
    p.add_argument("--feature-ratio", type=float, default=0.20)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--out", type=Path, default=Path("phase3_formal_10seed"))
    return p.parse_args()


def main():
    a = parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    runs_path = a.out / "runs.csv"
    history_path = a.out / "history.csv"
    if runs_path.exists():
        run_rows = pd.read_csv(runs_path).to_dict("records")
    else:
        run_rows = []
    if history_path.exists():
        history_rows = pd.read_csv(history_path).to_dict("records")
    else:
        history_rows = []
    completed = {(str(r["dataset"]), int(r["seed"]), str(r["method"])) for r in run_rows}

    for ds in a.datasets:
        bundle = load_highdim_dataset(ds, a.data_dir)
        for seed in a.seeds:
            Xtr, Xte, ytr, yte = train_test_split(
                bundle.X, bundle.y, test_size=0.30, stratify=bundle.y, random_state=seed
            )
            envs = make_nested_ir_environments(Xtr, ytr, [1, 3, 5], 3, seed)
            stable_scores, stable_info = cross_ir_stable_relevance(envs, a.gamma)
            base_pop = make_base_population(a.pop, bundle.X.shape[1], seed, a.feature_ratio)
            perm_scores = np.random.default_rng(seed + 9001).permutation(stable_scores)

            init_map = {
                "mean": base_pop.copy(),
                "stable25": matched_guidance(
                    base_pop, stable_scores, seed, a.guided_fraction, a.feature_ratio
                ),
                "permuted_control": matched_guidance(
                    base_pop, perm_scores, seed, a.guided_fraction, a.feature_ratio
                ),
            }

            for method in ["mean", "permuted_control", "stable25"]:
                if (ds, int(seed), method) in completed:
                    continue
                evaluator = FeatureSubsetEvaluator(envs, mode="mean", n_neighbors=5, backend="sklearn")
                t0 = time.perf_counter()
                result = run_nsga2(
                    evaluator.evaluate,
                    bundle.X.shape[1],
                    population_size=a.pop,
                    generations=a.gen,
                    random_state=seed,
                    verbose=False,
                    initial_population=init_map[method],
                )
                wall = time.perf_counter() - t0

                rep_idx = select_ideal_distance(result.pareto_objectives)
                rep = result.pareto_population[rep_idx]
                tg = heldout_gmeans(rep, envs, Xte, yte, 5)
                vals = np.asarray([tg[ir] for ir in sorted(tg)], dtype=float)
                front_test = test_front_metrics(result.pareto_population, envs, Xte, yte, 5)

                run_rows.append({
                    "dataset": ds,
                    "seed": seed,
                    "method": method,
                    "n_samples": int(bundle.X.shape[0]),
                    "n_features": int(bundle.X.shape[1]),
                    "selected_features": int(rep.sum()),
                    "feature_ratio": float(rep.mean()),
                    "test_mean_gmean": float(vals.mean()),
                    "test_worst_gmean": float(vals.min()),
                    "test_std_gmean": float(vals.std()),
                    "train_hv": float(hypervolume_2d(result.pareto_objectives)),
                    "unique_evaluations": int(len(evaluator.cache)),
                    "wall_seconds": float(wall),
                    **front_test,
                })
                for rec in result.history:
                    history_rows.append({
                        "dataset": ds,
                        "seed": seed,
                        "method": method,
                        "generation": int(rec["generation"]),
                        "unique_evaluations": rec.get("unique_evaluations"),
                        "train_hv": rec.get("train_hypervolume_ref_1_1"),
                        "pareto_size": int(rec["pareto_size"]),
                    })
                completed.add((ds, int(seed), method))
                pd.DataFrame(run_rows).to_csv(runs_path, index=False)
                pd.DataFrame(history_rows).to_csv(history_path, index=False)
                print(ds, seed, method, run_rows[-1], flush=True)

    runs = pd.DataFrame(run_rows)
    history = pd.DataFrame(history_rows)
    runs.to_csv(a.out / "runs.csv", index=False)
    history.to_csv(a.out / "history.csv", index=False)

    summary = runs.groupby(["dataset", "method"], as_index=False).agg(
        test_mean_gmean_mean=("test_mean_gmean", "mean"),
        test_mean_gmean_std=("test_mean_gmean", "std"),
        test_worst_gmean_mean=("test_worst_gmean", "mean"),
        test_worst_gmean_std=("test_worst_gmean", "std"),
        train_hv_mean=("train_hv", "mean"),
        train_hv_std=("train_hv", "std"),
        test_pf_hv_mean=("test_pf_hv", "mean"),
        test_pf_hv_std=("test_pf_hv", "std"),
        selected_features_mean=("selected_features", "mean"),
        selected_features_std=("selected_features", "std"),
        unique_evaluations_mean=("unique_evaluations", "mean"),
        wall_seconds_mean=("wall_seconds", "mean"),
    )
    summary.to_csv(a.out / "summary.csv", index=False)
    tests = paired_tests(runs)
    tests.to_csv(a.out / "paired_tests.csv", index=False)


if __name__ == "__main__":
    main()
