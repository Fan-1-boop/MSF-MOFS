#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读取真实完整的40组配对实验，拒绝不完整的正式图。"""
from __future__ import annotations
import argparse,json,sys,subprocess
from pathlib import Path
import pandas as pd
DATASETS=['madelon','hill_valley','musk1','arcene']
METHODS=['MSF-MOFS','HFMOEA-CP','MBLPE-CP']

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--allow-partial',action='store_true',help='只生成明确标记为探索版的结果')
    a=p.parse_args();base=a.out.resolve()
    missing=[]; inputs=[]
    for ds in DATASETS:
        for seed in range(92,102):
            fp=base/'by_seed'/ds/str(seed)/'per_generation_hv.csv'
            if not fp.exists(): missing.append(f'{ds}/{seed}'); continue
            try:
                d=pd.read_csv(fp,encoding='utf-8-sig')
                if len(d)!=579 or any(d[d.method.eq(m)].generation.tolist()!=list(range(193)) for m in METHODS):
                    missing.append(f'{ds}/{seed}')
                else: inputs.append(fp)
            except Exception:missing.append(f'{ds}/{seed}')
    print(f'完整配对：{len(inputs)}/40，尚缺：{missing}')
    if missing and not a.allow_partial:
        raise SystemExit('正式图拒绝生成：未完成40组配对。可先添加 --allow-partial 查看实际已完成结果。')
    if not inputs:raise SystemExit('没有数据')
    out=base/'analysis';out.mkdir(exist_ok=True)
    cmd=[sys.executable,str(Path(__file__).parent/'merge_analyze_pop30.py'), '--inputs', *map(str,inputs),'--out',str(out)]
    if not missing:cmd.append('--require-complete')
    subprocess.run(cmd,check=True)
    fig=base/'figure';fig.mkdir(exist_ok=True)
    subprocess.run([sys.executable,str(Path(__file__).parent/'plot_pop30_pilot.py'),
                    '--input',str(out/'per_generation_hv.csv'),'--out',str(fig)],check=True)
    print('图表与分析均来自实际记录，完成比例：',len(inputs),'/40')
if __name__=='__main__':main()
