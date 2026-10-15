#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""打包可独立续跑、可校验的完整配对结果，绝不夹带进行中的 CSV。"""
from __future__ import annotations
from pathlib import Path
import hashlib,json,shutil,sys,tempfile,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_missing import DATASETS, SEEDS, complete
ROOT=Path(__file__).resolve().parents[1]
DST=Path('/mnt/data/MSF_MOFS_Pop30_G192_Continue_20261014.zip')
with tempfile.TemporaryDirectory(prefix='msf_pop30_checkpoint_') as temporary:
    T=Path(temporary); c=0
    for fname in ['README_中文.md','run_all_windows.ps1']:
        shutil.copy2(ROOT/fname,T/fname)
    (T/'scripts').mkdir()
    for p in (ROOT/'scripts').glob('*.py'):
        shutil.copy2(p,T/'scripts'/p.name)
    source=Path('/mnt/data/population30_convergence/delivery/raw/pop30_algorithm_equivalence_6gens.json')
    (T/'evidence').mkdir()
    shutil.copy2(source,T/'evidence'/source.name)
    manifest=[]
    for ds in DATASETS:
        for seed in SEEDS:
            p=ROOT/'by_seed'/ds/str(seed)/'per_generation_hv.csv'
            if not complete(p,ds,seed): continue
            rel=Path('by_seed')/ds/str(seed)
            q=T/rel;q.mkdir(parents=True)
            shutil.copy2(p,q/p.name)
            origin=ROOT/'by_seed'/ds/str(seed)/'origin.json'
            if origin.exists():shutil.copy2(origin,q/'origin.json')
            manifest.append({'dataset':ds,'seed':seed,'path':(rel/'per_generation_hv.csv').as_posix(),
                             'sha256':hashlib.sha256(p.read_bytes()).hexdigest()});c+=1
    (T/'completion_manifest.json').write_text(json.dumps({'completed_triplets':c,'target_triplets':40,'completed_algorithms':c*3,'target_algorithms':120,'records':manifest},ensure_ascii=False,indent=2),encoding='utf-8')
    with (T/'SHA256SUMS.txt').open('w',encoding='utf-8') as out:
        for p in sorted(T.rglob('*')):
            if p.is_file() and p.name!='SHA256SUMS.txt':
                out.write(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(T).as_posix()+'\n')
    with zipfile.ZipFile(DST,'w',zipfile.ZIP_DEFLATED,compresslevel=7) as archive:
        for p in sorted(T.rglob('*')):
            if p.is_file():archive.write(p,p.relative_to(T).as_posix())
    with zipfile.ZipFile(DST) as archive:
        assert archive.testzip() is None
        for line in archive.read('SHA256SUMS.txt').decode('utf-8').splitlines():
            hash_,rel=line.split('  ',1)
            assert hashlib.sha256(archive.read(rel)).hexdigest()==hash_
print('已打包：',DST,'completed',c,'/40; bytes',DST.stat().st_size)
