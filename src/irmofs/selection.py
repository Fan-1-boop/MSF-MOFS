# -*- coding: utf-8 -*-
# MSF-MOFS 正式复现源码。
# 中文阅读说明：本文件保留原版所有可执行语句与随机数调用顺序；本次仅增加中文注释。
# 修改优化器、目标函数或评估协议之前，必须重新进行完整论文结果回归。
from __future__ import annotations

import numpy as np


def select_ideal_distance(F: np.ndarray) -> int:
    """Select by normalized Euclidean distance to the Pareto ideal point."""
    F = np.asarray(F, dtype=float)
    if len(F) == 0:
        raise ValueError("Empty Pareto front")
    fmin = F.min(axis=0)
    fmax = F.max(axis=0)
    scale = np.where(fmax > fmin, fmax - fmin, 1.0)
    normalized = (F - fmin) / scale
    return int(np.argmin(np.linalg.norm(normalized, axis=1)))


# 从帕累托候选集中根据拐点准则选择解。
def select_knee(F: np.ndarray) -> int:
    """Select the point farthest from the chord between the two objective extremes."""
    F = np.asarray(F, dtype=float)
    if len(F) <= 2:
        return select_ideal_distance(F)
    fmin = F.min(axis=0)
    fmax = F.max(axis=0)
    scale = np.where(fmax > fmin, fmax - fmin, 1.0)
    P = (F - fmin) / scale
    ia = int(np.argmin(P[:, 0]))
    ib = int(np.argmin(P[:, 1]))
    a, b = P[ia], P[ib]
    v = b - a
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return select_ideal_distance(F)
    d = np.abs(v[0] * (a[1] - P[:, 1]) - (a[0] - P[:, 0]) * v[1]) / nv
    d[[ia, ib]] = -np.inf
    if np.all(~np.isfinite(d)):
        return select_ideal_distance(F)
    return int(np.argmax(d))


def select_near_best_sparse(F: np.ndarray, tolerance: float = 0.02) -> int:
    """Sparsest Pareto point within `tolerance` of the best training mean G-mean.

    F[:,0] is assumed to be 1 - mean training G-mean and F[:,1] feature ratio.
    The default 0.02 is an additive two-percentage-point G-mean tolerance.
    """
    F = np.asarray(F, dtype=float)
    if len(F) == 0:
        raise ValueError("Empty Pareto front")
    g = 1.0 - F[:, 0]
    best = float(g.max())
    eligible = np.flatnonzero(g >= best - float(tolerance) - 1e-12)
    return int(sorted(eligible.tolist(), key=lambda i: (F[i, 1], -g[i]))[0])


def select_performance_first(F: np.ndarray) -> int:
    """Highest training mean G-mean; ties are broken by feature sparsity."""
    F = np.asarray(F, dtype=float)
    if len(F) == 0:
        raise ValueError("Empty Pareto front")
    g = 1.0 - F[:, 0]
    best = float(g.max())
    eligible = np.flatnonzero(np.isclose(g, best, atol=1e-12))
    return int(min(eligible.tolist(), key=lambda i: F[i, 1]))
