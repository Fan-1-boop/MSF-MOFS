# -*- coding: utf-8 -*-
# MSF-MOFS 正式复现源码。
# 中文阅读说明：本文件保留原版所有可执行语句与随机数调用顺序；本次仅增加中文注释。
# 修改优化器、目标函数或评估协议之前，必须重新进行完整论文结果回归。
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import io
import zipfile

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder


@dataclass
class DatasetBundle:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]


def _clean_xy(X, y) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    if isinstance(X, pd.DataFrame):
        feature_names = [str(c) for c in X.columns]
        X_df = X.copy()
    else:
        X_arr = np.asarray(X)
        feature_names = [f"f{i}" for i in range(X_arr.shape[1])]
        X_df = pd.DataFrame(X_arr, columns=feature_names)

    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    if X_df.isna().any().any():
        raise ValueError(
            "Dataset contains missing/non-numeric values. Phase 1 avoids global imputation to prevent "
            "pre-split leakage; add fold-local imputation before using such a dataset."
        )
    X_arr = X_df.to_numpy(dtype=float)

    y_arr = np.asarray(y).reshape(-1)
    if y_arr.dtype.kind not in "biufc":
        y_arr = LabelEncoder().fit_transform(y_arr.astype(str))
    else:
        y_arr = LabelEncoder().fit_transform(y_arr)

    classes = np.unique(y_arr)
    if len(classes) != 2:
        raise ValueError(f"Phase 1 currently supports binary classification only; got {len(classes)} classes.")

    return X_arr.astype(float), y_arr.astype(int), feature_names


def _read_member_from_zip(path: Path, preferred_names: list[str]) -> io.BytesIO:
    with zipfile.ZipFile(path, "r") as zf:
        members = zf.namelist()
        lower_to_member = {Path(m).name.lower(): m for m in members if not m.endswith("/")}
        for name in preferred_names:
            member = lower_to_member.get(name.lower())
            if member is not None:
                return io.BytesIO(zf.read(member))
    raise ValueError(f"Could not find any of {preferred_names} inside {path}")


def _load_ionosphere_local(path: Path) -> DatasetBundle:
    # Official UCI file has no header; some mirrors add a 35-column header row.
    source = _read_member_from_zip(path, ["ionosphere.data", "ionosphere_csv.csv"]) if path.suffix.lower() == ".zip" else path
    df = pd.read_csv(source, header=None)
    if df.shape[1] == 35:
        first_numeric = pd.to_numeric(df.iloc[0, :34], errors="coerce")
        if first_numeric.isna().any():
            if hasattr(source, "seek"):
                source.seek(0)
            df = pd.read_csv(source)
    if df.shape[1] != 35:
        raise ValueError(f"Ionosphere must have 35 columns (34 features + label); got {df.shape[1]} from {path}")
    X, y, names = _clean_xy(df.iloc[:, :34], df.iloc[:, 34])
    names = [f"Attribute{i}" for i in range(1, 35)]
    return DatasetBundle("ionosphere", X, y, names)


def _load_qsar_local(path: Path) -> DatasetBundle:
    # Official UCI file is semicolon-separated, no header, 41 descriptors + class.
    source = _read_member_from_zip(path, ["biodeg.csv"]) if path.suffix.lower() == ".zip" else path
    df = pd.read_csv(source, sep=";", header=None)
    if df.shape[1] != 42:
        # Permit a comma-separated mirror if the user provides one.
        if hasattr(source, "seek"):
            source.seek(0)
        df = pd.read_csv(source, header=None)
    if df.shape[1] != 42:
        raise ValueError(f"QSAR biodegradation must have 42 columns (41 features + label); got {df.shape[1]} from {path}")
    X, y, names = _clean_xy(df.iloc[:, :41], df.iloc[:, 41])
    names = [f"descriptor_{i}" for i in range(1, 42)]
    return DatasetBundle("qsar", X, y, names)


def _find_local_dataset(key: str, data_dir: Path) -> Path | None:
    data_dir = Path(data_dir)
    candidates = {
        "ionosphere": ["ionosphere.data", "ionosphere.csv", "ionosphere_csv.csv", "ionosphere.zip"],
        "qsar": [
            "biodeg.csv",
            "qsar_biodegradation.csv",
            "qsar biodegradation.zip",
            "qsar_biodegradation.zip",
            "qsar+biodegradation.zip",
        ],
    }[key]
    for name in candidates:
        path = data_dir / name
        if path.exists():
            return path
    return None


def load_dataset(name: str, data_dir: str | Path | None = None) -> DatasetBundle:
    key = name.lower().strip()

    if key in {"wdbc", "breast_cancer", "breast-cancer"}:
        ds = load_breast_cancer(as_frame=True)
        X, y, names = _clean_xy(ds.data, ds.target)
        return DatasetBundle("wdbc", X, y, names)

    aliases = {
        "ionosphere": "ionosphere",
        "qsar": "qsar",
        "qsar_biodegradation": "qsar",
    }
    if key not in aliases:
        raise ValueError(f"Unknown dataset '{name}'. Supported: wdbc, ionosphere, qsar")
    canonical = aliases[key]

    # Prefer an explicitly supplied local copy. This makes formal experiments reproducible
    # and avoids depending on network availability at run time.
    if data_dir is not None:
        local = _find_local_dataset(canonical, Path(data_dir))
        if local is not None:
            return _load_ionosphere_local(local) if canonical == "ionosphere" else _load_qsar_local(local)

    # Fall back to the official UCI Python interface when internet access is available.
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        location = f" under {Path(data_dir).resolve()}" if data_dir is not None else ""
        raise RuntimeError(
            f"No local {canonical} dataset found{location}, and ucimlrepo is not installed. "
            "Place the official UCI file in --data-dir or install ucimlrepo on an internet-enabled machine."
        ) from exc

    uci_id = 52 if canonical == "ionosphere" else 254
    ds = fetch_ucirepo(id=uci_id)
    X, y, names = _clean_xy(ds.data.features, ds.data.targets)
    return DatasetBundle(canonical, X, y, names)
