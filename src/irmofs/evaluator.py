# -*- coding: utf-8 -*-
# MSF-MOFS 正式复现源码。
# 中文阅读说明：本文件保留原版所有可执行语句与随机数调用顺序；本次仅增加中文注释。
# 修改优化器、目标函数或评估协议之前，必须重新进行完整论文结果回归。
# 建议配合 docs/03_MSF-MOFS算法原理.md 阅读核心数学定义和方法对应关系。
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .environments import IREnvironment


# 二分类 G-mean：以灵敏度和特异度的几何平均衡量性能。
def binary_gmean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_labels = np.unique(y_true)
    if len(true_labels) != 2:
        raise ValueError("G-mean requires two classes in y_true.")
    cm = confusion_matrix(y_true, y_pred, labels=true_labels)
    recalls = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    return float(np.sqrt(np.prod(recalls)))


def _make_knn(n_neighbors: int, n_train: int) -> Pipeline:
    k = min(n_neighbors, n_train)
    return Pipeline([
        ("scale", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])


@dataclass(frozen=True)
class _FastFold:
    # 在每个训练折独立标准化后，预计算各特征对欧氏距离平方的贡献。
    # 张量形状为 [验证样本数, 训练样本数, 特征数]。
    diff2: np.ndarray
    train_y: np.ndarray
    valid_y: np.ndarray


def _prepare_fast_folds(env: IREnvironment) -> tuple[_FastFold, ...]:
    folds: list[_FastFold] = []
    for tr, va in env.folds:
        xtr = np.asarray(env.X[tr], dtype=np.float64)
        xva = np.asarray(env.X[va], dtype=np.float64)
        mean = xtr.mean(axis=0)
        scale = xtr.std(axis=0, ddof=0)
        scale = np.where(scale > 0, scale, 1.0)
        xtr = (xtr - mean) / scale
        xva = (xva - mean) / scale
        # 用 float32 降低内存占用，同时尽量保持邻居排序稳定；
        # 评估染色体时，只需汇总所选特征的距离平方贡献，
        # 无需重新计算所有特征的样本距离。
        diff = xva[:, None, :] - xtr[None, :, :]
        diff2 = np.square(diff, dtype=np.float64)
        folds.append(_FastFold(diff2=diff2, train_y=env.y[tr].copy(), valid_y=env.y[va].copy()))
    return tuple(folds)


def _fast_knn_predict(fold: _FastFold, mask: np.ndarray, n_neighbors: int) -> np.ndarray:
    # 欧氏距离与距离平方具有相同排序，因此不必额外开方。
    dist2 = fold.diff2[:, :, mask].sum(axis=2, dtype=np.float64)
    k = min(int(n_neighbors), dist2.shape[1])
    nn = np.argsort(dist2, axis=1, kind="stable")[:, :k]
    neighbor_y = fold.train_y[nn]
    classes = np.unique(fold.train_y)
    # 与 sklearn 一致：类别票数相同时，argmax 优先选择排序靠前的类别。
    votes = np.stack([(neighbor_y == c).sum(axis=1) for c in classes], axis=1)
    return classes[np.argmax(votes, axis=1)]


@dataclass
class FeatureSubsetEvaluator:
    environments: dict[float, IREnvironment]
    mode: str = "single"
    target_ir: float = 3.0
    n_neighbors: int = 5
    backend: str = "sklearn"
    softmin_tau: float = 3.0
    stability_lambda: float = 0.5
    reference_gmeans: dict[float, float] | None = None
    prepared_fast_folds: dict[float, tuple[_FastFold, ...]] | None = None
    cache: dict[bytes, np.ndarray] = field(default_factory=dict)
    details_cache: dict[bytes, dict] = field(default_factory=dict)
    _fast_folds: dict[float, tuple[_FastFold, ...]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        backend = self.backend.lower()
        if self.softmin_tau <= 0:
            raise ValueError("softmin_tau must be > 0")
        if backend not in {"fast", "sklearn"}:
            raise ValueError("backend must be 'fast' or 'sklearn'")
        self.backend = backend
        if backend == "fast":
            if self.prepared_fast_folds is not None:
                self._fast_folds = self.prepared_fast_folds
            else:
                self._fast_folds = {ir: _prepare_fast_folds(env) for ir, env in self.environments.items()}

    def _gmean_for_environment(self, mask: np.ndarray, env: IREnvironment) -> float:
        fold_scores = []
        if self.backend == "fast":
            for fold in self._fast_folds[env.ir]:
                pred = _fast_knn_predict(fold, mask, self.n_neighbors)
                fold_scores.append(binary_gmean(fold.valid_y, pred))
        else:
            X = env.X[:, mask]
            for tr, va in env.folds:
                pipe = _make_knn(self.n_neighbors, len(tr))
                pipe.fit(X[tr], env.y[tr])
                pred = pipe.predict(X[va])
                fold_scores.append(binary_gmean(env.y[va], pred))
        return float(np.mean(fold_scores))

    def evaluate(self, chromosome: np.ndarray) -> np.ndarray:
        bits = np.asarray(chromosome, dtype=np.uint8)
        if bits.ndim != 1:
            raise ValueError("Chromosome must be one-dimensional.")
        key = bits.tobytes()
        if key in self.cache:
            return self.cache[key].copy()

        mask = bits.astype(bool)
        mode = self.mode.lower()
        if mask.sum() == 0:
            obj = np.array([1.0, 1.0, 1.0]) if mode == "robust" else np.array([1.0, 1.0])
            self.cache[key] = obj
            self.details_cache[key] = {"gmeans": {}, "n_features": 0}
            return obj.copy()

        if mode == "single":
            if self.target_ir not in self.environments:
                raise ValueError(
                    f"target_ir={self.target_ir} is not present in environments {sorted(self.environments)}"
                )
            env = self.environments[self.target_ir]
            g = self._gmean_for_environment(mask, env)
            gmeans = {self.target_ir: g}
            obj = np.array([1.0 - g, float(mask.mean())])
        elif mode in {"mean", "worst", "robust", "softmin", "regret_mean", "regret_max", "range_penalty"}:
            gmeans = {ir: self._gmean_for_environment(mask, env) for ir, env in self.environments.items()}
            vals = np.array(list(gmeans.values()), dtype=float)
            if mode == "mean":
                obj = np.array([1.0 - vals.mean(), float(mask.mean())])
            elif mode == "softmin":
                tau = float(self.softmin_tau)
                # 风险敏感的平滑最小值：对于 [0,1] 的 G-mean，该值介于
                # 算术均值（τ趋近0）和严格最小值之间。
                a = -tau * vals
                amax = float(np.max(a))
                risk_gmean = -(amax + np.log(np.mean(np.exp(a - amax)))) / tau
                obj = np.array([1.0 - risk_gmean, float(mask.mean())])
            elif mode == "worst":
                obj = np.array([1.0 - vals.min(), float(mask.mean())])
            elif mode == "range_penalty":
                lam = float(self.stability_lambda)
                robust_score = float(vals.mean() - lam * (vals.max() - vals.min()))
                obj = np.array([1.0 - robust_score, float(mask.mean())])
            elif mode in {"regret_mean", "regret_max"}:
                if not self.reference_gmeans:
                    raise ValueError(f"mode={mode} requires reference_gmeans")
                refs = np.array([float(self.reference_gmeans[ir]) for ir in gmeans], dtype=float)
                denom = np.maximum(np.abs(refs), 1e-8)
                regrets = (refs - vals) / denom
                regret = float(regrets.mean()) if mode == "regret_mean" else float(regrets.max())
                obj = np.array([regret, float(mask.mean())])
            else:
                obj = np.array([1.0 - vals.mean(), 1.0 - vals.min(), float(mask.mean())])
        else:
            raise ValueError("mode must be one of: single, mean, softmin, worst, robust, regret_mean, regret_max, range_penalty")

        vals = np.array(list(gmeans.values()), dtype=float)
        self.cache[key] = obj
        self.details_cache[key] = {
            "gmeans": gmeans,
            "mean_gmean": float(vals.mean()),
            "min_gmean": float(vals.min()),
            "std_gmean": float(vals.std()),
            "range_gmean": float(vals.max() - vals.min()),
            "range_penalized_gmean": float(vals.mean() - float(self.stability_lambda) * (vals.max() - vals.min())),
            "softmin_gmean": float(-(np.max(-float(self.softmin_tau) * vals) + np.log(np.mean(np.exp((-float(self.softmin_tau) * vals) - np.max(-float(self.softmin_tau) * vals))))) / float(self.softmin_tau)),
            "reference_gmeans": None if not self.reference_gmeans else {float(k): float(v) for k, v in self.reference_gmeans.items()},
            "regrets": None if not self.reference_gmeans else {
                float(ir): float((float(self.reference_gmeans[ir]) - gmeans[ir]) / max(abs(float(self.reference_gmeans[ir])), 1e-8))
                for ir in gmeans
            },
            "n_features": int(mask.sum()),
            "feature_ratio": float(mask.mean()),
        }
        return obj.copy()

    def details(self, chromosome: np.ndarray) -> dict:
        bits = np.asarray(chromosome, dtype=np.uint8)
        key = bits.tobytes()
        if key not in self.details_cache:
            self.evaluate(bits)
        return dict(self.details_cache[key])


# 确认阶段使用独立保留数据评估已固定的解，不参与搜索。
def heldout_gmeans(
    chromosome: np.ndarray,
    train_environments: dict[float, IREnvironment],
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_neighbors: int = 5,
) -> dict[float, float]:
    """Evaluate IR-specific training conditions on one common untouched outer test set.

    A separate scaler + KNN is fitted on each IR-specific outer-training environment.
    Every fitted model is then evaluated on exactly the same outer test samples. This
    isolates the effect of the *training* imbalance ratio and avoids injecting extra
    variance by resampling the held-out test set for each IR.
    """
    mask = np.asarray(chromosome, dtype=np.uint8).astype(bool)
    if mask.sum() == 0:
        return {ir: 0.0 for ir in train_environments}

    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    scores: dict[float, float] = {}
    for ir in sorted(train_environments):
        tr_env = train_environments[ir]
        model = _make_knn(n_neighbors, len(tr_env.y))
        model.fit(tr_env.X[:, mask], tr_env.y)
        pred = model.predict(X_test[:, mask])
        scores[ir] = binary_gmean(y_test, pred)
    return scores
