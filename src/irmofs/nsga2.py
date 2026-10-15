# -*- coding: utf-8 -*-
# MSF-MOFS 正式复现源码。
# 中文阅读说明：本文件保留原版所有可执行语句与随机数调用顺序；本次仅增加中文注释。
# 修改优化器、目标函数或评估协议之前，必须重新进行完整论文结果回归。
# 建议配合 docs/03_MSF-MOFS算法原理.md 阅读核心数学定义和方法对应关系。
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class NSGA2Result:
    population: np.ndarray
    objectives: np.ndarray
    ranks: np.ndarray
    crowding: np.ndarray
    history: list[dict]

    @property
    def pareto_mask(self) -> np.ndarray:
        return self.ranks == 0

    @property
    def pareto_population(self) -> np.ndarray:
        return self.population[self.pareto_mask]

    @property
    def pareto_objectives(self) -> np.ndarray:
        return self.objectives[self.pareto_mask]


# 双目标帕累托支配关系，用于比较两个候选特征子集。
def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


# 非支配分层：先按支配关系划分等级。
def fast_non_dominated_sort(F: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
    n = len(F)
    dominated = [[] for _ in range(n)]
    domination_count = np.zeros(n, dtype=int)
    rank = np.full(n, -1, dtype=int)
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(p + 1, n):
            if dominates(F[p], F[q]):
                dominated[p].append(q)
                domination_count[q] += 1
            elif dominates(F[q], F[p]):
                dominated[q].append(p)
                domination_count[p] += 1

    for p in range(n):
        if domination_count[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front: list[int] = []
        for p in fronts[i]:
            for q in dominated[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        i += 1

    return fronts, rank


# 拥挤距离：在同层个体之间维持多样性。
def crowding_distance(F: np.ndarray, fronts: list[list[int]]) -> np.ndarray:
    distances = np.zeros(len(F), dtype=float)
    for front in fronts:
        if not front:
            continue
        if len(front) <= 2:
            distances[front] = np.inf
            continue
        front_idx = np.asarray(front, dtype=int)
        values = F[front_idx]
        for m in range(F.shape[1]):
            order = np.argsort(values[:, m], kind="mergesort")
            sorted_idx = front_idx[order]
            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf
            vmin = values[order[0], m]
            vmax = values[order[-1], m]
            if vmax <= vmin:
                continue
            for j in range(1, len(order) - 1):
                idx = sorted_idx[j]
                if np.isinf(distances[idx]):
                    continue
                prev_v = values[order[j - 1], m]
                next_v = values[order[j + 1], m]
                distances[idx] += (next_v - prev_v) / (vmax - vmin)
    return distances


def _tournament(rng: np.random.Generator, ranks: np.ndarray, crowding: np.ndarray) -> int:
    a, b = rng.integers(0, len(ranks), size=2)
    if ranks[a] < ranks[b]:
        return int(a)
    if ranks[b] < ranks[a]:
        return int(b)
    if crowding[a] > crowding[b]:
        return int(a)
    if crowding[b] > crowding[a]:
        return int(b)
    return int(a if rng.random() < 0.5 else b)


def _ensure_nonempty(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if x.sum() == 0:
        x[rng.integers(0, len(x))] = 1
    return x


def _standard_mutate(c: np.ndarray, rng: np.random.Generator, mutation_prob: float) -> np.ndarray:
    mut = rng.random(len(c)) < mutation_prob
    c[mut] = 1 - c[mut]
    return c


def _agm_mutate(
    c: np.ndarray,
    rng: np.random.Generator,
    mutation_prob: float,
    feature_frequency: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Archive-guided adaptive bit-flip mutation.

    For an unselected feature j, the inclusion probability is increased when j is
    frequent in the external nondominated archive. For a selected feature, the
    deletion probability is increased when j is rare in the archive. The expected
    scale remains close to the standard mutation rate and the guidance strength is
    controlled by alpha. alpha=0 exactly recovers ordinary bit-flip mutation.
    """
    freq = np.asarray(feature_frequency, dtype=float)
    if freq.shape != c.shape:
        raise ValueError("feature_frequency shape must match chromosome shape")
    alpha = float(np.clip(alpha, 0.0, 1.0))

    # 乘子限制在 [1-α, 1+α]，避免档案信息导致某些特征被确定性锁死；
    # 同时保证变异预算仍与对照方法可比较。
    add_prob = mutation_prob * ((1.0 - alpha) + 2.0 * alpha * freq)
    del_prob = mutation_prob * ((1.0 - alpha) + 2.0 * alpha * (1.0 - freq))
    probs = np.where(c == 0, add_prob, del_prob)
    probs = np.clip(probs, 0.0, 1.0)
    mut = rng.random(len(c)) < probs
    c[mut] = 1 - c[mut]
    return c


# 产生子代；随机数调用顺序影响可复现性，切勿随意调整。
def _offspring(
    population: np.ndarray,
    ranks: np.ndarray,
    crowding: np.ndarray,
    rng: np.random.Generator,
    crossover_prob: float,
    mutation_prob: float,
    mutation_strategy: str = "standard",
    feature_frequency: np.ndarray | None = None,
    alpha: float = 0.0,
) -> np.ndarray:
    n, d = population.shape
    children: list[np.ndarray] = []
    while len(children) < n:
        p1 = population[_tournament(rng, ranks, crowding)]
        p2 = population[_tournament(rng, ranks, crowding)]
        c1, c2 = p1.copy(), p2.copy()
        if rng.random() < crossover_prob:
            mask = rng.random(d) < 0.5
            c1 = np.where(mask, p1, p2).astype(np.uint8)
            c2 = np.where(mask, p2, p1).astype(np.uint8)
        for c in (c1, c2):
            if mutation_strategy == "standard":
                _standard_mutate(c, rng, mutation_prob)
            elif mutation_strategy == "agm":
                if feature_frequency is None:
                    raise ValueError("AGM requires feature_frequency")
                _agm_mutate(c, rng, mutation_prob, feature_frequency, alpha)
            else:
                raise ValueError("mutation_strategy must be 'standard' or 'agm'")
            children.append(_ensure_nonempty(c, rng))
            if len(children) >= n:
                break
    return np.asarray(children, dtype=np.uint8)


def _environmental_select(pop: np.ndarray, F: np.ndarray, n_keep: int):
    # 环境选择前移除基因型完全相同的个体，避免重复占据种群位置。
    _, unique_idx = np.unique(pop, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    pop = pop[unique_idx]
    F = F[unique_idx]

    fronts, ranks = fast_non_dominated_sort(F)
    crowd = crowding_distance(F, fronts)

    chosen: list[int] = []
    for front in fronts:
        if len(chosen) + len(front) <= n_keep:
            chosen.extend(front)
        else:
            need = n_keep - len(chosen)
            if need > 0:
                front_sorted = sorted(front, key=lambda i: crowd[i], reverse=True)
                chosen.extend(front_sorted[:need])
            break

    idx = np.asarray(chosen, dtype=int)
    pop2, F2 = pop[idx], F[idx]
    fronts2, ranks2 = fast_non_dominated_sort(F2)
    crowd2 = crowding_distance(F2, fronts2)
    return pop2, F2, ranks2, crowd2


def _update_archive(
    archive_pop: np.ndarray | None,
    archive_F: np.ndarray | None,
    pop: np.ndarray,
    F: np.ndarray,
    max_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Maintain an external archive of historically nondominated unique solutions."""
    if archive_pop is None or archive_F is None or len(archive_pop) == 0:
        combined_pop = pop.copy()
        combined_F = F.copy()
    else:
        combined_pop = np.vstack([archive_pop, pop])
        combined_F = np.vstack([archive_F, F])

    _, unique_idx = np.unique(combined_pop, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    combined_pop = combined_pop[unique_idx]
    combined_F = combined_F[unique_idx]

    fronts, ranks = fast_non_dominated_sort(combined_F)
    pf_idx = np.flatnonzero(ranks == 0)
    a_pop = combined_pop[pf_idx]
    a_F = combined_F[pf_idx]

    if len(a_pop) > max_size:
        pf_fronts, _ = fast_non_dominated_sort(a_F)
        crowd = crowding_distance(a_F, pf_fronts)
        # 保留边界极端解以及目标空间内相对孤立的内部解。
        order = np.argsort(-crowd, kind="stable")[:max_size]
        a_pop = a_pop[order]
        a_F = a_F[order]
    return a_pop, a_F


def _archive_feature_frequency(archive_pop: np.ndarray, smoothing: float = 1.0) -> np.ndarray:
    """Laplace-smoothed feature occurrence frequency in the external archive."""
    if len(archive_pop) == 0:
        raise ValueError("archive_pop must be non-empty")
    n = float(len(archive_pop))
    return (archive_pop.sum(axis=0, dtype=float) + smoothing) / (n + 2.0 * smoothing)



# 计算二维最小化帕累托集的超体积，参考点通常为 (1, 1)。
def hypervolume_2d(F: np.ndarray, ref: tuple[float, float] = (1.0, 1.0)) -> float:
    """Exact 2-D minimization hypervolume for a nondominated or mixed point set."""
    P = np.asarray(F, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2 or len(P) == 0:
        return 0.0
    P = P[(P[:, 0] <= ref[0]) & (P[:, 1] <= ref[1])]
    if len(P) == 0:
        return 0.0
    P = P[np.argsort(P[:, 0], kind="stable")]
    nd = []
    best_y = float(ref[1])
    for x, y in P:
        if y < best_y:
            nd.append((float(x), float(y)))
            best_y = float(y)
    hv = 0.0
    for i, (x, y) in enumerate(nd):
        x_next = nd[i + 1][0] if i + 1 < len(nd) else float(ref[0])
        hv += max(0.0, x_next - x) * max(0.0, float(ref[1]) - y)
    return float(hv)

# 主优化循环：Fisher 引导初始化后进行迭代演化。
def run_nsga2(
    evaluate: Callable[[np.ndarray], np.ndarray],
    n_features: int,
    population_size: int = 40,
    generations: int = 30,
    crossover_prob: float = 0.9,
    mutation_prob: float | None = None,
    init_feature_prob: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
    mutation_strategy: str = "standard",
    agm_alpha_max: float = 0.8,
    archive_max_size: int | None = None,
    initial_population: np.ndarray | None = None,
) -> NSGA2Result:
    if population_size < 4:
        raise ValueError("population_size must be >= 4")
    mutation_strategy = mutation_strategy.lower()
    if mutation_strategy not in {"standard", "agm"}:
        raise ValueError("mutation_strategy must be 'standard' or 'agm'")
    if not 0.0 <= agm_alpha_max <= 1.0:
        raise ValueError("agm_alpha_max must be in [0,1]")

    rng = np.random.default_rng(random_state)
    mutation_prob = (1.0 / n_features) if mutation_prob is None else mutation_prob
    archive_max_size = (2 * population_size) if archive_max_size is None else int(archive_max_size)

    if initial_population is None:
        pop = (rng.random((population_size, n_features)) < init_feature_prob).astype(np.uint8)
    else:
        pop = np.asarray(initial_population, dtype=np.uint8).copy()
        if pop.ndim != 2 or pop.shape[1] != n_features:
            raise ValueError("initial_population must have shape [n, n_features]")
        if len(pop) > population_size:
            pop = pop[:population_size]
        elif len(pop) < population_size:
            extra = (rng.random((population_size - len(pop), n_features)) < init_feature_prob).astype(np.uint8)
            pop = np.vstack([pop, extra])
    for i in range(population_size):
        _ensure_nonempty(pop[i], rng)
    F = np.vstack([evaluate(x) for x in pop])
    fronts, ranks = fast_non_dominated_sort(F)
    crowd = crowding_distance(F, fronts)

    archive_pop: np.ndarray | None = None
    archive_F: np.ndarray | None = None
    if mutation_strategy == "agm":
        archive_pop, archive_F = _update_archive(None, None, pop, F, archive_max_size)

    history: list[dict] = []
    pf0 = F[ranks == 0]
    owner0 = getattr(evaluate, "__self__", None)
    eval_count0 = len(getattr(owner0, "cache", {})) if owner0 is not None else None
    history.append({
        "generation": 0,
        "pareto_size": int(len(pf0)),
        "objective_min": pf0.min(axis=0).tolist(),
        "objective_mean": pf0.mean(axis=0).tolist(),
        "train_hypervolume_ref_1_1": hypervolume_2d(pf0) if pf0.shape[1] == 2 else None,
        "unique_evaluations": eval_count0,
        "mutation_strategy": mutation_strategy,
        "agm_alpha": 0.0,
        "archive_size": int(len(archive_pop)) if archive_pop is not None else 0,
    })
    for gen in range(1, generations + 1):
        while len(pop) < population_size:
            x = (rng.random(n_features) < init_feature_prob).astype(np.uint8)
            _ensure_nonempty(x, rng)
            pop = np.vstack([pop, x])
            F = np.vstack([F, evaluate(x)])
        fronts, ranks = fast_non_dominated_sort(F)
        crowd = crowding_distance(F, fronts)

        alpha = 0.0
        feature_frequency = None
        if mutation_strategy == "agm":
            assert archive_pop is not None
            # 演化早期保持弱引导，后期逐步增加利用已知优良特征的程度。
            alpha = agm_alpha_max * (gen / generations)
            feature_frequency = _archive_feature_frequency(archive_pop)

        child = _offspring(
            pop,
            ranks,
            crowd,
            rng,
            crossover_prob,
            mutation_prob,
            mutation_strategy=mutation_strategy,
            feature_frequency=feature_frequency,
            alpha=alpha,
        )
        child_F = np.vstack([evaluate(x) for x in child])
        combined_pop = np.vstack([pop, child])
        combined_F = np.vstack([F, child_F])
        pop, F, ranks, crowd = _environmental_select(combined_pop, combined_F, population_size)

        if mutation_strategy == "agm":
            archive_pop, archive_F = _update_archive(
                archive_pop, archive_F, pop, F, archive_max_size
            )

        pf = F[ranks == 0]
        owner = getattr(evaluate, "__self__", None)
        eval_count = len(getattr(owner, "cache", {})) if owner is not None else None
        record = {
            "generation": gen,
            "pareto_size": int(len(pf)),
            "objective_min": pf.min(axis=0).tolist(),
            "objective_mean": pf.mean(axis=0).tolist(),
            "train_hypervolume_ref_1_1": hypervolume_2d(pf) if pf.shape[1] == 2 else None,
            "unique_evaluations": eval_count,
            "mutation_strategy": mutation_strategy,
            "agm_alpha": float(alpha),
            "archive_size": int(len(archive_pop)) if archive_pop is not None else 0,
        }
        history.append(record)
        if verbose:
            mins = ", ".join(f"{v:.4f}" for v in record["objective_min"])
            extra = f" alpha={alpha:.3f} archive={record['archive_size']}" if mutation_strategy == "agm" else ""
            print(f"gen={gen:03d} pareto={len(pf):03d} min_obj=[{mins}]{extra}")

    return NSGA2Result(pop, F, ranks, crowd, history)
