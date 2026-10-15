# -*- coding: utf-8 -*-
# MSF-MOFS 正式复现源码。
# 中文阅读说明：本文件保留原版所有可执行语句与随机数调用顺序；本次仅增加中文注释。
# 修改优化器、目标函数或评估协议之前，必须重新进行完整论文结果回归。
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class IREnvironment:
    ir: float
    X: np.ndarray
    y: np.ndarray
    folds: tuple[tuple[np.ndarray, np.ndarray], ...]


def _majority_minority_labels(y: np.ndarray) -> tuple[int, int]:
    labels, counts = np.unique(y, return_counts=True)
    if len(labels) != 2:
        raise ValueError("IR environment construction requires binary labels.")
    # On an exact class-count tie, argmax and argmin both return index 0.
    # Keep the two classes distinct so balanced datasets can be converted into
    # controlled IR environments correctly.
    if counts[0] == counts[1]:
        majority, minority = int(labels[0]), int(labels[1])
    else:
        majority = int(labels[np.argmax(counts)])
        minority = int(labels[np.argmin(counts)])
    return majority, minority


# 根据预先固定的随机种子构造训练环境和交叉验证折，防止数据泄漏。
def make_nested_ir_environments(
    X: np.ndarray,
    y: np.ndarray,
    irs: Iterable[float] = (1, 3, 5),
    n_splits: int = 3,
    random_state: int = 42,
) -> dict[float, IREnvironment]:
    """Create controlled, nested class-imbalance environments.

    A common majority subset is fixed across all IRs. Minority subsets are nested:
    minority(IR_high) subset minority(IR_low). This isolates class-ratio change as much
    as possible from sample-identity change.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    irs = tuple(sorted(float(ir) for ir in irs))
    if any(ir < 1 for ir in irs):
        raise ValueError("All imbalance ratios must be >= 1.")

    maj_label, min_label = _majority_minority_labels(y)
    maj_idx = np.flatnonzero(y == maj_label)
    min_idx = np.flatnonzero(y == min_label)

    rng = np.random.default_rng(random_state)
    maj_perm = rng.permutation(maj_idx)
    min_perm = rng.permutation(min_idx)

    # A fixed majority pool must also fit the balanced IR=1 environment.
    n_majority = min(len(maj_perm), len(min_perm))
    majority_common = maj_perm[:n_majority]

    envs: dict[float, IREnvironment] = {}
    for i, ir in enumerate(irs):
        n_minority = max(n_splits, int(np.floor(n_majority / ir)))
        n_minority = min(n_minority, len(min_perm))
        minority_subset = min_perm[:n_minority]
        idx = np.concatenate([majority_common, minority_subset])
        idx = rng.permutation(idx)
        X_env = X[idx]
        y_env = y[idx]

        _, class_counts = np.unique(y_env, return_counts=True)
        if class_counts.min() < n_splits:
            raise ValueError(
                f"IR={ir:g} has only {class_counts.min()} minority samples, fewer than n_splits={n_splits}."
            )

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state + 1009 * (i + 1),
        )
        folds = tuple((tr, va) for tr, va in skf.split(X_env, y_env))
        envs[ir] = IREnvironment(ir=ir, X=X_env, y=y_env, folds=folds)

    return envs


def describe_environments(envs: dict[float, IREnvironment]) -> list[dict]:
    rows = []
    for ir, env in sorted(envs.items()):
        labels, counts = np.unique(env.y, return_counts=True)
        rows.append(
            {
                "ir": ir,
                "n_samples": int(len(env.y)),
                "class_counts": {int(k): int(v) for k, v in zip(labels, counts)},
                "realized_ir": float(counts.max() / counts.min()),
            }
        )
    return rows
