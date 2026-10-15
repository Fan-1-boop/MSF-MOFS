#!/usr/bin/env python3
"""MSF-MOFS search core used by the satellite feature-selection validation.

This file contains only feature-selection/search logic. It intentionally excludes
all satellite health estimation, evidence fusion, smoothing, DBN, thresholds, and
deployment-model training.

MSF construction: one Fisher ranking, S(D)={10,0.05D,0.10D,0.20D}, three
smallest distinct guided cardinalities, exactly three guided chromosomes, and
standard NSGA-II thereafter.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

META = {
    "scenarioId", "split", "tick", "state", "label", "faultScope", "faultKey", "faultPair", "faultMode",
    "magnitude", "startTick", "readiness",
}


def gmean_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tnr = tn / max(1, tn + fp)
    tpr = tp / max(1, tp + fn)
    return float(math.sqrt(max(0.0, tnr * tpr)))


def balanced_subsample(X: np.ndarray, y: np.ndarray, limit: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(y) <= limit:
        return X, y
    rng = np.random.default_rng(seed)
    chosen: List[int] = []
    per_class = limit // 2
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        take = min(len(idx), per_class)
        chosen.extend(rng.choice(idx, size=take, replace=False).tolist())
    remaining = limit - len(chosen)
    if remaining > 0:
        pool = np.setdiff1d(np.arange(len(y)), np.asarray(chosen, dtype=int), assume_unique=False)
        if len(pool):
            chosen.extend(rng.choice(pool, size=min(remaining, len(pool)), replace=False).tolist())
    rng.shuffle(chosen)
    idx = np.asarray(chosen, dtype=int)
    return X[idx], y[idx]


def fisher_scores(X: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    pos = X[y == 1]
    neg = X[y == 0]
    mu_p = np.mean(pos, axis=0)
    mu_n = np.mean(neg, axis=0)
    var_p = np.var(pos, axis=0)
    var_n = np.var(neg, axis=0)
    return (mu_p - mu_n) ** 2 / (var_p + var_n + eps)


def msf_cardinalities(D: int) -> List[int]:
    values = [10, int(round(0.05 * D)), int(round(0.10 * D)), int(round(0.20 * D))]
    values = sorted({max(1, min(D, value)) for value in values})
    if len(values) < 3:
        for value in range(1, D + 1):
            if value not in values:
                values.append(value)
            if len(values) >= 3:
                break
    return sorted(values)[:3]


@dataclass
class Individual:
    mask: np.ndarray
    obj: Tuple[float, float] | None = None
    rank: int = 0
    crowding: float = 0.0

    def key(self) -> bytes:
        return np.packbits(self.mask.astype(np.uint8)).tobytes()


def dominates(a: Individual, b: Individual) -> bool:
    assert a.obj is not None and b.obj is not None
    return all(x <= y for x, y in zip(a.obj, b.obj)) and any(x < y for x, y in zip(a.obj, b.obj))


def fast_non_dominated_sort(pop: Sequence[Individual]) -> List[List[Individual]]:
    domination_sets: Dict[int, List[int]] = {i: [] for i in range(len(pop))}
    dominated_count = [0] * len(pop)
    fronts: List[List[int]] = [[]]
    for i, p in enumerate(pop):
        for j, q in enumerate(pop):
            if i == j:
                continue
            if dominates(p, q):
                domination_sets[i].append(j)
            elif dominates(q, p):
                dominated_count[i] += 1
        if dominated_count[i] == 0:
            p.rank = 0
            fronts[0].append(i)
    k = 0
    while k < len(fronts) and fronts[k]:
        nxt: List[int] = []
        for i in fronts[k]:
            for j in domination_sets[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    pop[j].rank = k + 1
                    nxt.append(j)
        if nxt:
            fronts.append(nxt)
        k += 1
    return [[pop[i] for i in front] for front in fronts if front]


def assign_crowding(front: Sequence[Individual]) -> None:
    if not front:
        return
    for item in front:
        item.crowding = 0.0
    if len(front) <= 2:
        for item in front:
            item.crowding = float("inf")
        return
    for m in range(2):
        ordered = sorted(front, key=lambda ind: ind.obj[m])
        ordered[0].crowding = ordered[-1].crowding = float("inf")
        low = ordered[0].obj[m]
        high = ordered[-1].obj[m]
        if high == low:
            continue
        for i in range(1, len(ordered) - 1):
            ordered[i].crowding += (ordered[i + 1].obj[m] - ordered[i - 1].obj[m]) / (high - low)


def rank_and_crowd(pop: Sequence[Individual]) -> List[List[Individual]]:
    fronts = fast_non_dominated_sort(pop)
    for front in fronts:
        assign_crowding(front)
    return fronts


def tournament(rng: random.Random, pop: Sequence[Individual]) -> Individual:
    a, b = rng.sample(list(pop), 2)
    if a.rank != b.rank:
        return a if a.rank < b.rank else b
    if a.crowding != b.crowding:
        return a if a.crowding > b.crowding else b
    return a if rng.random() < 0.5 else b


def evaluate_population(
    pop: Sequence[Individual], X: np.ndarray, y: np.ndarray, cache: Dict[bytes, Tuple[float, float]], seed: int,
) -> None:
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    D = X.shape[1]
    for ind in pop:
        key = ind.key()
        if key in cache:
            ind.obj = cache[key]
            continue
        cols = np.flatnonzero(ind.mask)
        if len(cols) == 0:
            obj = (1.0, 0.0)
        else:
            scores = []
            for train_idx, valid_idx in cv.split(X, y):
                model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
                model.fit(X[train_idx][:, cols], y[train_idx])
                pred = model.predict(X[valid_idx][:, cols])
                scores.append(gmean_score(y[valid_idx], pred))
            g = float(np.mean(scores))
            obj = (1.0 - g, len(cols) / D)
        cache[key] = obj
        ind.obj = obj


def make_initial_population(
    D: int, fisher_order: np.ndarray, rng: np.random.Generator, guided: bool, population_size: int = 12,
) -> List[Individual]:
    masks = rng.random((population_size, D)) < 0.20
    for row in masks:
        if not row.any():
            row[int(rng.integers(0, D))] = True
    if guided:
        for i, k in enumerate(msf_cardinalities(D)):
            masks[i, :] = False
            masks[i, fisher_order[:k]] = True
    return [Individual(mask=row.copy()) for row in masks]


def evolve(
    X: np.ndarray,
    y: np.ndarray,
    fisher_order: np.ndarray,
    seed: int,
    guided: bool = True,
    population_size: int = 12,
    generations: int = 6,
) -> Tuple[List[Individual], Dict[bytes, Tuple[float, float]]]:
    rng_np = np.random.default_rng(seed)
    rng = random.Random(seed)
    D = X.shape[1]
    pop = make_initial_population(D, fisher_order, rng_np, guided=guided, population_size=population_size)
    cache: Dict[bytes, Tuple[float, float]] = {}
    evaluate_population(pop, X, y, cache, seed)
    rank_and_crowd(pop)

    for generation in range(generations):
        offspring: List[Individual] = []
        while len(offspring) < population_size:
            p1 = tournament(rng, pop)
            p2 = tournament(rng, pop)
            c1 = p1.mask.copy()
            c2 = p2.mask.copy()
            if rng.random() < 0.9:
                selector = rng_np.random(D) < 0.5
                tmp = c1[selector].copy()
                c1[selector] = c2[selector]
                c2[selector] = tmp
            for child in (c1, c2):
                mutation = rng_np.random(D) < (1.0 / D)
                child ^= mutation
                if not child.any():
                    child[int(rng_np.integers(0, D))] = True
                offspring.append(Individual(mask=child))
                if len(offspring) >= population_size:
                    break
        evaluate_population(offspring, X, y, cache, seed + generation + 1)
        combined = pop + offspring
        fronts = rank_and_crowd(combined)
        next_pop: List[Individual] = []
        for front in fronts:
            if len(next_pop) + len(front) <= population_size:
                next_pop.extend(front)
            else:
                front_sorted = sorted(front, key=lambda ind: ind.crowding, reverse=True)
                next_pop.extend(front_sorted[: population_size - len(next_pop)])
                break
        pop = next_pop
        rank_and_crowd(pop)
    return pop, cache


def holdout_gmean(X_train, y_train, X_eval, y_eval, cols: np.ndarray) -> float:
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    model.fit(X_train[:, cols], y_train)
    pred = model.predict(X_eval[:, cols])
    return gmean_score(y_eval, pred)


def choose_from_pareto(
    pop: Sequence[Individual], X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
) -> Tuple[Individual, List[dict]]:
    fronts = rank_and_crowd(pop)
    pareto = fronts[0]
    records = []
    for ind in pareto:
        cols = np.flatnonzero(ind.mask)
        val_g = holdout_gmean(X_train, y_train, X_val, y_val, cols)
        records.append({"ind": ind, "cols": cols, "val_gmean": val_g})
    best_g = max(item["val_gmean"] for item in records)
    near_best = [item for item in records if item["val_gmean"] >= best_g - 0.005]
    chosen = min(near_best, key=lambda item: (len(item["cols"]), -item["val_gmean"]))
    serializable = [
        {
            "featureCount": int(len(item["cols"])),
            "cvGmean": float(1.0 - item["ind"].obj[0]),
            "valGmean": float(item["val_gmean"]),
        }
        for item in sorted(records, key=lambda item: (len(item["cols"]), -item["val_gmean"]))
    ]
    return chosen["ind"], serializable

