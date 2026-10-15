# -*- coding: utf-8 -*-
# MSF-MOFS 正式复现源码。
# 中文阅读说明：本文件保留原版所有可执行语句与随机数调用顺序；本次仅增加中文注释。
# 修改优化器、目标函数或评估协议之前，必须重新进行完整论文结果回归。
from __future__ import annotations
from pathlib import Path
import io, subprocess, tempfile, zipfile
import numpy as np
import pandas as pd
from .datasets import DatasetBundle, _clean_xy

def _member(zf, suffix):
    for name in zf.namelist():
        if name.lower().endswith(suffix.lower()):
            return zf.read(name)
    raise ValueError(f"Missing {suffix}")

def load_highdim_dataset(name: str, data_dir: str | Path) -> DatasetBundle:
    data_dir = Path(data_dir)
    key = name.lower().strip()

    if key == "madelon":
        path = data_dir / "madelon.zip"
        with zipfile.ZipFile(path) as zf:
            Xtr = pd.read_csv(io.BytesIO(_member(zf, "madelon_train.data")), sep=r"\s+", header=None)
            ytr = pd.read_csv(io.BytesIO(_member(zf, "madelon_train.labels")), header=None).iloc[:,0]
            Xva = pd.read_csv(io.BytesIO(_member(zf, "madelon_valid.data")), sep=r"\s+", header=None)
            yva = pd.read_csv(io.BytesIO(_member(zf, "madelon_valid.labels")), header=None).iloc[:,0]
        X, y, _ = _clean_xy(pd.concat([Xtr, Xva], ignore_index=True), pd.concat([ytr, yva], ignore_index=True))
        return DatasetBundle("madelon", X, y, [f"madelon_{i}" for i in range(1, 501)])

    if key in {"hill", "hill_valley"}:
        path = data_dir / "hill+valley.zip"
        with zipfile.ZipFile(path) as zf:
            tr = pd.read_csv(io.BytesIO(_member(zf, "Hill_Valley_without_noise_Training.data")))
            te = pd.read_csv(io.BytesIO(_member(zf, "Hill_Valley_without_noise_Testing.data")))
        df = pd.concat([tr, te], ignore_index=True)
        X, y, _ = _clean_xy(df.iloc[:,:-1], df.iloc[:,-1])
        return DatasetBundle("hill_valley", X, y, [f"X{i}" for i in range(1, 101)])

    if key in {"musk", "musk1"}:
        path = data_dir / "musk+version+1.zip"
        with zipfile.ZipFile(path) as zf:
            payload = _member(zf, "clean1.data.Z")
        # v2 portability-only change: bundle bytes decoded from the SAME archive member.
        # This avoids requiring Unix `uncompress` on Windows. SHA-256 provenance in
        # data/public/core/MUSK1_DECOMPRESSION.json; historical loader in legacy/.
        decoded = data_dir / "musk1_clean1_decompressed.data"
        if decoded.is_file():
            payload = decoded.read_bytes()
        else:
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / "clean1.data.Z"
                p.write_bytes(payload)
                payload = subprocess.check_output(["uncompress", "-c", str(p)])
        df = pd.read_csv(io.BytesIO(payload), header=None)
        X, y, _ = _clean_xy(df.iloc[:,2:-1], df.iloc[:,-1])
        return DatasetBundle("musk1", X, y, [f"musk_{i}" for i in range(1, 167)])

    if key == "arcene":
        path = data_dir / "arcene.zip"
        with zipfile.ZipFile(path) as zf:
            Xtr = pd.read_csv(io.BytesIO(_member(zf, "arcene_train.data")), sep=r"\s+", header=None)
            ytr = pd.read_csv(io.BytesIO(_member(zf, "arcene_train.labels")), header=None).iloc[:, 0]
            Xva = pd.read_csv(io.BytesIO(_member(zf, "arcene_valid.data")), sep=r"\s+", header=None)
            # In the official UCI archive the validation labels are stored at the ZIP root.
            yva = pd.read_csv(io.BytesIO(_member(zf, "arcene_valid.labels")), header=None).iloc[:, 0]
        X, y, _ = _clean_xy(
            pd.concat([Xtr, Xva], ignore_index=True),
            pd.concat([ytr, yva], ignore_index=True),
        )
        return DatasetBundle("arcene", X, y, [f"arcene_{i}" for i in range(1, 10001)])

    raise ValueError("Supported high-dimensional datasets: madelon, hill_valley, musk1, arcene")
