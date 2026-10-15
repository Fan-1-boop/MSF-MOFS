# -*- coding: utf-8 -*-
# MSF-MOFS 正式复现源码。
# 中文阅读说明：本文件保留原版所有可执行语句与随机数调用顺序；本次仅增加中文注释。
# 修改优化器、目标函数或评估协议之前，必须重新进行完整论文结果回归。
# 建议配合 docs/03_MSF-MOFS算法原理.md 阅读核心数学定义和方法对应关系。
from __future__ import annotations

import numpy as np
from .environments import IREnvironment


# Fisher 分数：仅从训练数据统计类间与类内离散程度，禁止使用测试集标签。
def fisher_scores(X: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    labels = np.unique(y)
    if len(labels) != 2:
        raise ValueError('binary labels required')
    a = np.asarray(X[y == labels[0]], dtype=float)
    b = np.asarray(X[y == labels[1]], dtype=float)
    num = np.square(a.mean(axis=0) - b.mean(axis=0))
    den = a.var(axis=0, ddof=0) + b.var(axis=0, ddof=0) + eps
    return num / den


def _rank01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind='stable')
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    if len(x) <= 1:
        return np.ones_like(x, dtype=float)
    return ranks / (len(x) - 1)


# 跨不平衡率环境稳定性：汇总各环境的特征重要性以构造稳定排序。
def cross_ir_stable_relevance(
    environments: dict[float, IREnvironment],
    stability_gamma: float = 0.5,
) -> tuple[np.ndarray, dict]:
    per_ir = []
    irs = sorted(environments)
    for ir in irs:
        env = environments[ir]
        per_ir.append(_rank01(fisher_scores(env.X, env.y)))
    M = np.vstack(per_ir)
    mean = M.mean(axis=0)
    std = M.std(axis=0)
    stable = mean - float(stability_gamma) * std
    # 将特征重要性归一化到 [0, 1]，用于概率式引导。
    lo, hi = float(stable.min()), float(stable.max())
    norm = (stable - lo) / (hi - lo) if hi > lo else np.full_like(stable, 0.5)
    return norm, {
        'irs': [float(i) for i in irs],
        'mean_rank': mean,
        'std_rank': std,
        'raw_stable_score': stable,
        'normalized_stable_score': norm,
    }


# 根据训练集特征排序构造引导初始种群。
def make_guided_population(
    scores: np.ndarray,
    population_size: int,
    random_state: int,
    guided_fraction: float = 0.5,
    expected_feature_ratio: float = 0.2,
) -> np.ndarray:
    """Create a diverse mixed population using cross-IR stable relevance.

    Guided individuals use score-proportional Bernoulli probabilities whose average
    equals ``expected_feature_ratio``. The remaining individuals are standard random
    sparse chromosomes. A few deterministic top-k seeds are injected when possible.
    """
    s = np.asarray(scores, dtype=float)
    d = len(s)
    if d == 0:
        raise ValueError('empty score vector')
    if not 0 <= guided_fraction <= 1:
        raise ValueError('guided_fraction must be in [0,1]')
    rng = np.random.default_rng(random_state)
    n_guided = int(round(population_size * guided_fraction))
    n_random = population_size - n_guided

    # 通过正值仿射权重并缩放到目标期望稀疏度。
    w = 0.25 + 1.5 * s
    probs = expected_feature_ratio * w / max(float(w.mean()), 1e-12)
    probs = np.clip(probs, 0.01, 0.75)
    guided = (rng.random((n_guided, d)) < probs).astype(np.uint8) if n_guided else np.empty((0,d),dtype=np.uint8)
    random = (rng.random((n_random, d)) < expected_feature_ratio).astype(np.uint8) if n_random else np.empty((0,d),dtype=np.uint8)
    pop = np.vstack([guided, random])

    # 在多个稀疏层次生成确定性的高相关性初始染色体。
    if n_guided:
        order = np.argsort(-s, kind='stable')
        ks = sorted(set(max(1, int(round(d * r))) for r in (0.05, 0.10, 0.20)))
        for row, k in enumerate(ks[:n_guided]):
            pop[row] = 0
            pop[row, order[:k]] = 1

    for i in range(len(pop)):
        if pop[i].sum() == 0:
            pop[i, int(np.argmax(s))] = 1
    return pop


# 多尺度初始化：只替换预定数量的引导染色体，剩余个体保留随机基线。
def make_cisr_multiscale_population(
    base_population: np.ndarray,
    stable_scores: np.ndarray,
    guided_fraction: float = 0.25,
) -> tuple[np.ndarray, list[int]]:
    """Frozen CISR-MOFS initialization used after external-baseline development.

    With N=12 and guided_fraction=0.25, three matched-random individuals are
    replaced. Candidate sparsity sizes are {10, 5%D, 10%D, 20%D}; the three
    smallest unique valid sizes are used. This retains an absolute ultra-sparse
    seed for very high-dimensional data while preserving relative scales.

    Examples:
      D=100   -> [5, 10, 20]
      D=166   -> [8, 10, 17]
      D=500   -> [10, 25, 50]
      D=10000 -> [10, 500, 1000]
    """
    pop = np.asarray(base_population, dtype=np.uint8).copy()
    n, d = pop.shape
    n_guided = int(round(n * float(guided_fraction)))
    if n_guided <= 0:
        return pop, []
    order = np.argsort(-np.asarray(stable_scores, dtype=float), kind="stable")
    candidates = [
        10,
        max(1, int(round(0.05 * d))),
        max(1, int(round(0.10 * d))),
        max(1, int(round(0.20 * d))),
    ]
    sizes = sorted(set(min(d, k) for k in candidates))[:n_guided]
    for row, k in enumerate(sizes):
        pop[row] = 0
        pop[row, order[:k]] = 1
    return pop, sizes
