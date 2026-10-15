"""Checked merging of independent, already-completed search output shards.

No new experiment is created: this only deduplicates equal primary-key rows and
records checksums of every input. Conflicting result rows cause a hard error.
"""
from pathlib import Path
import hashlib
import json
import os
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def portable_locator(file: Path) -> dict:
    """来源定位仅存项目相对路径；外部文件不保存可能失效的绝对路径。"""
    try:
        return {"path": file.resolve().relative_to(PROJECT_ROOT).as_posix(), "path_kind": "project_relative"}
    except ValueError:
        return {"path": None, "file_name": file.name, "path_kind": "external_file_not_packaged"}

import pandas as pd
import numpy as np

KEY=['dataset','seed','carrier','init']
METRICS=['train_hv','test_pf_hv','unique_wrapper_evaluations',
         'best_test_mean_gmean_on_pf','min_features_on_pf','n_features']


def merge_core_inputs(inputs:list[Path], target:Path) -> dict:
    if len(inputs)<2:
        raise ValueError('Need at least two raw CSV inputs to merge')
    frames=[]
    for p in inputs:
        if not p.is_file():
            raise FileNotFoundError(p)
        frame=pd.read_csv(p)
        if not set(KEY+METRICS).issubset(frame.columns):
            raise ValueError(f'Schema mismatch in {p}')
        if frame.duplicated(KEY).any():
            raise ValueError(f'Duplicate keys within a single input {p}')
        frames.append(frame)
    df=pd.concat(frames,ignore_index=True,sort=False)
    identical_duplicates=0
    for key, group in df.groupby(KEY,dropna=False):
        if len(group)>1:
            values=group[METRICS].to_numpy(dtype=float)
            if not np.allclose(values,values[0:1,:],atol=1e-11,rtol=0):
                raise ValueError(f'Conflicting archived/rerun values for key={key}')
            identical_duplicates += len(group)-1
    merged=df.drop_duplicates(KEY,keep='first').sort_values(KEY,kind='stable').reset_index(drop=True)
    target.parent.mkdir(parents=True,exist_ok=True)
    # Record all pre-merge bytes BEFORE any file replacement. When source ==
    # destination, save a durable independent copy for later SHA verification.
    source_entries=[]
    for p,frame in zip(inputs,frames):
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        stable_path=p
        if p.resolve()==target.resolve():
            backup=target.parent/'merge_input_snapshots'
            backup.mkdir(parents=True,exist_ok=True)
            stable_path=backup/f'input_{digest[:24]}.csv'
            if not stable_path.exists():
                stable_path.write_bytes(p.read_bytes())
        source_entries.append({**portable_locator(stable_path),'sha256':digest,'rows':len(frame)})

    # Preserve previously written raw output when any input validation fails.
    with tempfile.NamedTemporaryFile('w',dir=target.parent,suffix='.csv',delete=False,encoding='utf-8',newline='') as tmp:
        temp_path=Path(tmp.name)
        merged.to_csv(tmp,index=False)
    os.replace(temp_path,target)
    manifest={'status':'MERGED_WITH_VERIFIED_DUPLICATE_KEYS',
              'n_input_rows':int(sum(map(len,frames))), 'deduplicated_rows':identical_duplicates,
              'n_final_rows':len(merged),'target':portable_locator(target),
              'inputs':source_entries,
              'not_new_search':True}
    (target.parent/'merge_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return manifest
