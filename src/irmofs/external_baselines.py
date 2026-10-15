# -*- coding: utf-8 -*-
# MSF-MOFS 正式复现源码。
# 中文阅读说明：本文件保留原版所有可执行语句与随机数调用顺序；本次仅增加中文注释。
# 修改优化器、目标函数或评估协议之前，必须重新进行完整论文结果回归。

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import os

import numpy as np
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.feature_selection._mutual_info import _compute_mi, _iterate_columns
from sklearn.preprocessing import MinMaxScaler, scale
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y
from scipy.stats import rankdata

from .nsga2 import (
    fast_non_dominated_sort,
    crowding_distance,
    _environmental_select,
    hypervolume_2d,
)
from .stable_guidance import fisher_scores


@dataclass
class ExternalResult:
    population: np.ndarray
    objectives: np.ndarray
    history: list[dict]

    @property
    def pareto_mask(self) -> np.ndarray:
        _, ranks = fast_non_dominated_sort(self.objectives)
        return ranks == 0

    @property
    def pareto_population(self) -> np.ndarray:
        return self.population[self.pareto_mask]

    @property
    def pareto_objectives(self) -> np.ndarray:
        return self.objectives[self.pareto_mask]


def _ensure_nonempty(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(x, dtype=np.uint8).copy()
    if x.sum() == 0:
        x[rng.integers(0, len(x))] = 1
    return x


def _rank_desc(scores: np.ndarray) -> np.ndarray:
    scores = np.nan_to_num(np.asarray(scores, float), nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def _binary_relief_scores(X: np.ndarray, y: np.ndarray, max_samples: int = 120) -> np.ndarray:
    """Deterministic nearest-hit/nearest-miss Relief-style score.

    A capped training-only sample is used for tractability, and the pairwise
    squared-distance matrix is computed once using a Gram formulation.
    """
    X = np.asarray(X, float)
    y = np.asarray(y)
    n = len(y)
    if n > max_samples:
        idx = np.linspace(0, n - 1, max_samples, dtype=int)
        Xs, ys = X[idx], y[idx]
    else:
        Xs, ys = X, y
    lo = np.nanmin(Xs, axis=0)
    hi = np.nanmax(Xs, axis=0)
    den = np.where(hi > lo, hi - lo, 1.0)
    Z = (Xs - lo) / den
    sq = np.sum(Z * Z, axis=1)
    D = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
    D = np.maximum(D, 0.0)
    np.fill_diagonal(D, np.inf)
    hits = np.empty(len(ys), dtype=int)
    misses = np.empty(len(ys), dtype=int)
    for i in range(len(ys)):
        same = np.flatnonzero(ys == ys[i])
        same = same[same != i]
        diff = np.flatnonzero(ys != ys[i])
        hits[i] = same[np.argmin(D[i, same])]
        misses[i] = diff[np.argmin(D[i, diff])]
    return np.mean(np.abs(Z - Z[misses]) - np.abs(Z - Z[hits]), axis=0)


_MI_X = None
_MI_Y = None

def _mi_chunk_worker(indices):
    X = _MI_X
    y = _MI_Y
    return [(int(j), float(_compute_mi(X[:, int(j)], y, False, True, 3))) for j in indices]

def _mutual_info_classif_parallel_dense(X: np.ndarray, y: np.ndarray, random_state: int) -> np.ndarray:
    """复现 sklearn 的连续特征互信息计算，兼容 Windows/Linux/macOS。

    数据预处理、随机扰动和逐列互信息公式保持原实现不变。
    能使用 fork 时继续采用历史进程池；Windows 不支持 fork，改用线程池。
    线程共享相同的 X/y，不依赖子进程继承全局变量，也不会改变特征排序。
    """
    global _MI_X, _MI_Y
    X, y = check_X_y(X, y, accept_sparse=False, y_numeric=False)
    X = X.astype(np.float64, copy=True)
    X = scale(X, with_mean=False, copy=False)
    rng = check_random_state(random_state)
    means = np.maximum(1, np.mean(np.abs(X), axis=0))
    X += 1e-10 * means * rng.standard_normal(size=X.shape)
    _MI_X, _MI_Y = X, y
    n_features = X.shape[1]
    workers = min(8, max(1, os.cpu_count() or 1), n_features)
    chunks = [a for a in np.array_split(np.arange(n_features), workers) if len(a)]
    out = np.empty(n_features, dtype=float)
    # Windows 的 multiprocessing 不提供 fork：原版 get_context("fork") 会直接报错。
    # 此处只更换并行执行后端；预处理、随机噪声和每列 _compute_mi 完全不变。
    # 相较于 spawn 子进程，线程共享已有 X/y，避免高维数据在多个进程间重复复制。
    try:
        if "fork" in mp.get_all_start_methods():
            executor = ProcessPoolExecutor(
                max_workers=workers, mp_context=mp.get_context("fork")
            )
        else:
            executor = ThreadPoolExecutor(max_workers=workers)
        with executor as ex:
            for part in ex.map(_mi_chunk_worker, chunks):
                for j, v in part:
                    out[j] = v
        return out
    finally:
        # 运行结束或异常时都释放数组引用，避免后续种子重复计算时占用额外内存。
        _MI_X, _MI_Y = None, None


def _hfmoea_filter_rankings(X: np.ndarray, y: np.ndarray, random_state: int) -> list[np.ndarray]:
    """Ten filter rankings matching the categories in the official HFMOEA code.
    The search protocol is adapted to the common CISR evaluator, but population
    guidance follows the published ten-filter initialization idea.
    """
    X = np.asarray(X, float)
    y = np.asarray(y)
    d = X.shape[1]
    eps = 1e-12

    # 1) Mutual information
    mi = _mutual_info_classif_parallel_dense(X, y, random_state)

    # 2) Spearman absolute correlation (vectorized via column ranks)
    RX = rankdata(X, axis=0)
    Ry = rankdata(y.astype(float))
    RXc = RX - RX.mean(axis=0)
    Ryc = Ry - Ry.mean()
    sden = np.sqrt((RXc * RXc).sum(axis=0) * np.sum(Ryc * Ryc)) + eps
    scc = np.abs((RXc * Ryc[:, None]).sum(axis=0) / sden)

    # 3) Relief-style
    relief = _binary_relief_scores(X, y)

    # 4) Pearson absolute correlation
    y0 = (y == np.unique(y)[1]).astype(float)
    Xc = X - X.mean(axis=0)
    yc = y0 - y0.mean()
    denom = np.sqrt((Xc * Xc).sum(axis=0) * (yc * yc).sum()) + eps
    pcc = np.abs((Xc * yc[:, None]).sum(axis=0) / denom)

    # 5) Chi-square after train-only min-max transform
    Xnn = MinMaxScaler().fit_transform(X)
    chi = chi2(Xnn, y)[0]
    chi = np.nan_to_num(chi, nan=0.0, posinf=np.finfo(float).max)

    # 6) Information gain (use a separately seeded MI estimate)
    info = _mutual_info_classif_parallel_dense(X, y, random_state + 137)

    # 7) Mean absolute deviation
    mad = np.mean(np.abs(X - X.mean(axis=0)), axis=0)

    # 8) Dispersion ratio (between-class mean separation / within-class spread)
    labs = np.unique(y)
    a, b = X[y == labs[0]], X[y == labs[1]]
    disp = np.abs(a.mean(axis=0) - b.mean(axis=0)) / (a.std(axis=0) + b.std(axis=0) + eps)

    # 9) Similarity/discriminability proxy: standardized centroid separation
    sim = (a.mean(axis=0) - b.mean(axis=0)) ** 2 / (X.var(axis=0) + eps)

    # 10) Fisher score
    fisher = fisher_scores(X, y)

    return [_rank_desc(v) for v in (mi, scc, relief, pcc, chi, info, mad, disp, sim, fisher)]


def run_hfmoea_adapted(
    evaluate: Callable[[np.ndarray], np.ndarray],
    X_reference: np.ndarray,
    y_reference: np.ndarray,
    n_features: int,
    population_size: int,
    generations: int,
    random_state: int,
    topk: int = 10,
    mutation_percent: float = 6.0,
    precomputed_rankings: list[np.ndarray] | None = None,
) -> ExternalResult:
    """Common-protocol adaptation of HFMOEA (Kundu & Mallipeddi, 2022).

    Published mechanism retained:
      * ten filter rankings create ten guided chromosomes;
      * remaining initial chromosomes are random;
      * binary crossover/mutation evolve the population.

    For fair comparison, all candidate subsets are scored by the exact same
    multi-IR wrapper evaluator used by CISR-MOFS.
    """
    if population_size < 10:
        raise ValueError("HFMOEA requires population_size >= 10")
    rng = np.random.default_rng(random_state)
    rankings = (
        _hfmoea_filter_rankings(X_reference, y_reference, random_state)
        if precomputed_rankings is None
        else precomputed_rankings
    )
    pop = np.zeros((population_size, n_features), dtype=np.uint8)
    k = min(max(1, int(topk)), n_features)
    for i, ranks in enumerate(rankings[:10]):
        pop[i, np.flatnonzero(ranks <= k)] = 1
    if population_size > 10:
        pop[10:] = rng.integers(0, 2, size=(population_size - 10, n_features), dtype=np.uint8)
    for i in range(population_size):
        pop[i] = _ensure_nonempty(pop[i], rng)

    F = np.vstack([evaluate(x) for x in pop])
    history = [{
        "generation": 0,
        "train_hv": hypervolume_2d(F[fast_non_dominated_sort(F)[1] == 0]),
    }]

    # Use standard NSGA-II elitist environmental selection, which is the intended
    # optimizer in the paper, while reproducing the official midpoint crossover
    # and shared mutation-coordinate mechanism.
    for gen in range(1, generations + 1):
        midpoint = n_features // 2
        children = np.empty_like(pop)
        for i in range(population_size):
            p1 = pop[i % population_size]
            p2 = pop[(i + 1) % population_size]
            children[i, :midpoint] = p1[:midpoint]
            children[i, midpoint:] = p2[midpoint:]

        num_mut = max(1, int(population_size * n_features * mutation_percent / 100.0))
        # Official code draws a common set of mutation coordinates and applies
        # them to every offspring. Cap to d unique coordinates.
        # Official code samples mutation coordinates with replacement and applies
        # the same coordinate list to every offspring. Repeated coordinates collapse
        # to one effective advanced-index assignment, so use the unique sampled set.
        mut_idx = np.unique(rng.integers(0, n_features, size=num_mut))
        children[:, mut_idx] = 1 - children[:, mut_idx]
        for i in range(population_size):
            children[i] = _ensure_nonempty(children[i], rng)

        CF = np.vstack([evaluate(x) for x in children])
        pop, F, _, _ = _environmental_select(
            np.vstack([pop, children]),
            np.vstack([F, CF]),
            population_size,
        )
        pf = F[fast_non_dominated_sort(F)[1] == 0]
        history.append({"generation": gen, "train_hv": hypervolume_2d(pf)})

    return ExternalResult(pop, F, history)


def _mblpe_fisher_prefilter(X: np.ndarray, population_size: int) -> np.ndarray:
    d = X.shape[1]
    T = min(d, population_size * 3)
    # Caller supplies binary y indirectly via a closure in run_mblpe_adapted;
    # this helper only returns T once scores are provided.
    return np.arange(T)


def _cal_mblpe_fitness(F: np.ndarray, pop: np.ndarray) -> np.ndarray:
    """Port of the official CalFitness.m: strength raw fitness + objective/decision density."""
    F = np.asarray(F, float)
    pop = np.asarray(pop, np.uint8)
    n = len(F)
    dominate = np.zeros((n, n), dtype=bool)
    for i in range(n - 1):
        for j in range(i + 1, n):
            k = int(np.any(F[i] < F[j])) - int(np.any(F[i] > F[j]))
            if k == 1:
                dominate[i, j] = True
            elif k == -1:
                dominate[j, i] = True
    S = dominate.sum(axis=1)
    R = np.array([S[dominate[:, i]].sum() for i in range(n)], dtype=float)

    def kth_density(D):
        D = np.asarray(D, dtype=float).copy()
        np.fill_diagonal(D, np.inf)
        D.sort(axis=1)
        kk = max(0, min(D.shape[1] - 1, int(np.floor(np.sqrt(n))) - 1))
        return 1.0 / (D[:, kk] + 2.0)

    do = np.sqrt(((F[:, None, :] - F[None, :, :]) ** 2).sum(axis=2))
    dd = np.abs(pop[:, None, :].astype(int) - pop[None, :, :].astype(int)).sum(axis=2)
    return R + kth_density(do) + kth_density(dd)


def _mblpe_environmental_select(pop: np.ndarray, F: np.ndarray, n_keep: int) -> tuple[np.ndarray, np.ndarray]:
    """Decision/objective duplicate handling inspired by official EnvironmentalSelection.m."""
    # exact genotype duplicates
    _, idx = np.unique(pop, axis=0, return_index=True)
    idx = np.sort(idx)
    pop, F = pop[idx], F[idx]

    fronts, ranks = fast_non_dominated_sort(F)
    ns_idx = np.flatnonzero(ranks == 0)
    ns_pop, ns_F = pop[ns_idx], F[ns_idx]

    kept_non_ns = []
    for front_id, front in enumerate(fronts[1:], start=1):
        if not front:
            continue
        front = np.asarray(front, int)
        # group by the sparsity objective (official code groups PopObj(:,2))
        vals = F[front, 1]
        for value in np.unique(vals):
            group = front[np.isclose(vals, value)]
            if len(group) == 1:
                kept_non_ns.append(int(group[0]))
                continue
            # choose the decision-space diverse member relative to the nearest
            # nondominated solution in objective space; tie-break with frequency.
            target_F = F[group[0]]
            if len(ns_idx):
                nearest = ns_idx[np.argmin(np.linalg.norm(ns_F - target_F, axis=1))]
                hamming = np.abs(pop[group].astype(int) - pop[nearest].astype(int)).sum(axis=1) / pop.shape[1]
            else:
                hamming = np.zeros(len(group))
            reference_idx = list(kept_non_ns) + ns_idx.tolist()
            if reference_idx:
                freq = pop[reference_idx].sum(axis=0)
                nu = np.array([freq[pop[g].astype(bool)].sum() for g in group], float)
            else:
                nu = np.zeros(len(group))
            # official sortrows([-ds1 nu]): maximize distance, then minimize nu
            order = np.lexsort((nu, -hamming))
            kept_non_ns.append(int(group[order[0]]))

    cand_idx = np.array(kept_non_ns + ns_idx.tolist(), dtype=int)
    if len(cand_idx) < n_keep:
        remaining = [i for i in range(len(pop)) if i not in set(cand_idx.tolist())]
        cand_idx = np.array(cand_idx.tolist() + remaining, dtype=int)
    cp, cF = pop[cand_idx], F[cand_idx]
    fit = _cal_mblpe_fitness(cF, cp)
    order = np.argsort(fit, kind="stable")[:min(n_keep, len(cp))]
    cp, cF = cp[order], cF[order]

    # MBLPE expects N survivors. If duplicate removal leaves fewer, refill by
    # repeating best unique survivors; this affects only mating history, not FES.
    if len(cp) < n_keep:
        reps = np.arange(n_keep) % len(cp)
        cp, cF = cp[reps].copy(), cF[reps].copy()
    return cp, cF


def run_mblpe_adapted(
    evaluate_full: Callable[[np.ndarray], np.ndarray],
    X_reference: np.ndarray,
    y_reference: np.ndarray,
    n_features_original: int,
    population_size: int,
    generations: int,
    random_state: int,
    threshold: float = 0.6,
    prefilter_selected: np.ndarray | None = None,
) -> ExternalResult:
    """Common-protocol port of MBLPE (Zhou et al., EAAI 2025).

    Retains the official high-dimensional Fisher prefilter (T=min(D,3N)),
    three-population DE warmup, linear-predictive reproduction, duplicate
    handling and feature-analysis environmental selection.

    Candidate masks are expanded back to the original feature space before
    being evaluated by the common multi-IR wrapper objective.
    """
    rng = np.random.default_rng(random_state)
    D = int(n_features_original)
    T = min(D, population_size * 3)
    if prefilter_selected is None:
        scores = fisher_scores(X_reference, y_reference)
        selected = np.argsort(-scores, kind="stable")[:T]
    else:
        selected = np.asarray(prefilter_selected, dtype=int)
        if len(selected) != T:
            raise ValueError(f"prefilter_selected must contain exactly T={T} features")
    d = len(selected)

    cache: dict[bytes, np.ndarray] = {}
    def eval_reduced(z: np.ndarray) -> np.ndarray:
        z = _ensure_nonempty(z, rng)
        key = z.tobytes()
        if key in cache:
            return cache[key].copy()
        full = np.zeros(D, dtype=np.uint8)
        full[selected] = z
        val = np.asarray(evaluate_full(full), float).copy()
        # The official algorithm searches in the Fisher-reduced space and uses
        # selected-count / reduced-dimension as its second internal objective.
        val[1] = float(z.mean())
        cache[key] = val
        return val.copy()

    Fscale, CR = 0.7, 0.7
    max_fes = population_size * generations

    # Official thresholded random initialization: P(bit=1) ~= 0.4.
    pop = (rng.random((population_size, d)) > threshold).astype(np.uint8)
    for i in range(population_size):
        pop[i] = _ensure_nonempty(pop[i], rng)
    F = np.vstack([eval_reduced(x) for x in pop])
    fitcount = population_size
    chain = [(pop.copy(), F.copy())]
    history = [{"generation": 1, "train_hv": hypervolume_2d(F[fast_non_dominated_sort(F)[1] == 0])}]

    # Create generations 2 and 3 via DE-style reproduction.
    for gen in (2, 3):
        trials = []
        for i in range(population_size):
            candidates = np.delete(np.arange(population_size), i)
            a, b, c = rng.choice(candidates, size=3, replace=False)
            mutant = pop[a].astype(float) + Fscale * (pop[b].astype(float) - pop[c].astype(float))
            bad = (mutant < 0) | (mutant > 1)
            mutant[bad] = rng.random(bad.sum())
            trial = pop[i].astype(float).copy()
            jrand = rng.integers(0, d)
            mask = rng.random(d) <= CR
            mask[jrand] = True
            trial[mask] = mutant[mask]
            trial = (trial > threshold).astype(np.uint8)
            trials.append(_ensure_nonempty(trial, rng))
        trials = np.unique(np.asarray(trials, np.uint8), axis=0)
        TF = np.vstack([eval_reduced(x) for x in trials])
        fitcount += len(trials)
        pop, F = _mblpe_environmental_select(
            np.vstack([trials, pop]),
            np.vstack([TF, F]),
            population_size,
        )
        chain.append((pop.copy(), F.copy()))
        history.append({"generation": gen, "train_hv": hypervolume_2d(F[fast_non_dominated_sort(F)[1] == 0])})
        if fitcount >= max_fes:
            break

    gen = len(chain) + 1
    while fitcount < max_fes and gen <= generations:
        p1, p2, p3 = chain[-3][0], chain[-2][0], chain[-1][0]
        i1, i2, i3 = rng.permutation(population_size), rng.permutation(population_size), rng.permutation(population_size)
        trial_cont = (4.0 * p3[i3].astype(float) + p2[i2].astype(float) - 2.0 * p1[i1].astype(float)) / 3.0
        bad = (trial_cont < 0) | (trial_cont > 1)
        trial_cont[bad] = rng.random(bad.sum())
        trials = (trial_cont > threshold).astype(np.uint8)
        for i in range(len(trials)):
            trials[i] = _ensure_nonempty(trials[i], rng)
        trials = np.unique(trials, axis=0)
        # remove genotypes seen in the three-generation history
        seen = {x.tobytes() for p, _ in chain[-3:] for x in p}
        trials = np.asarray([x for x in trials if x.tobytes() not in seen], dtype=np.uint8)
        if len(trials) == 0:
            # deterministic escape matching the intent of handle_decision_vectors
            trials = (rng.random((population_size, d)) > threshold).astype(np.uint8)
            for i in range(len(trials)):
                trials[i] = _ensure_nonempty(trials[i], rng)
            trials = np.unique(trials, axis=0)

        remaining = max_fes - fitcount
        trials = trials[:remaining]
        TF = np.vstack([eval_reduced(x) for x in trials])
        fitcount += len(trials)
        pop, F = _mblpe_environmental_select(
            np.vstack([trials, chain[-1][0]]),
            np.vstack([TF, chain[-1][1]]),
            population_size,
        )
        chain.append((pop.copy(), F.copy()))
        chain = chain[-3:]
        history.append({"generation": gen, "train_hv": hypervolume_2d(F[fast_non_dominated_sort(F)[1] == 0])})
        gen += 1

    # Expand the final population to the original space and restore the common
    # feature-ratio objective for cross-method HV reporting.
    full_pop = np.zeros((len(pop), D), dtype=np.uint8)
    full_pop[:, selected] = pop
    common_F = np.vstack([evaluate_full(x) for x in full_pop])
    return ExternalResult(full_pop, common_F, history)
