#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补跑四数据集 Population=30、192代实验，逐组存储并严格核对。

默认每个数据集-种子由独立子进程运行三种方法。程序不会重跑完整文件；
不会覆盖原始论文 Table 2 结果；子进程向不同目录写入，避免数据竞争。
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib, json, os, subprocess, sys, time, traceback
import pandas as pd

DATASETS=('hill_valley','musk1','arcene','madelon')
METHODS=('MSF-MOFS','HFMOEA-CP','MBLPE-CP')
SEEDS=tuple(range(92,102))
BASE=Path(__file__).resolve().parents[1]

def complete(csv: Path, dataset: str, seed: int) -> bool:
    if not csv.is_file(): return False
    try:
        df=pd.read_csv(csv,encoding='utf-8-sig')
        if len(df)!=579: return False
        if set(df['dataset'])!={dataset} or set(df['seed'])!={seed}: return False
        for method in METHODS:
            q=df[df.method.eq(method)]
            if q.generation.tolist()!=list(range(193)): return False
            if q[['test_pf_hv','train_pf_hv']].isna().any().any(): return False
            if (q.unique_wrapper_evaluations.diff().dropna()<0).any():return False
        return True
    except Exception:return False

def summary(output: Path, project_root: Path):
    got=[]; todo=[]
    for ds in DATASETS:
        for seed in SEEDS:
            csv=output/'by_seed'/ds/str(seed)/'per_generation_hv.csv'
            if complete(csv,ds,seed):
                got.append(dict(dataset=ds,seed=seed,rows=579,sha256=hashlib.sha256(csv.read_bytes()).hexdigest()))
            else:todo.append(dict(dataset=ds,seed=seed))
    status={'experiment':'Population=30, generation 0..192, three methods, four datasets',
            'completed_triplets':len(got),'target_triplets':40,'completed_algorithms':len(got)*3,
            'target_algorithms':120,'missing_triplets':todo,'completed':got,
            'project_root_at_run':str(project_root)}
    tmp=output/'progress.json.tmp';tmp.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8'); tmp.replace(output/'progress.json')
    return status

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=BASE)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--max-triplets',type=int,default=40)
    args=ap.parse_args()
    project=args.project_root.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    script=out/'scripts'/'run_generation_convergence.py'
    if not script.exists():raise SystemExit(f'缺少跟踪脚本：{script}')
    for x in ['src/irmofs/nsga2.py','src/irmofs/external_baselines.py','data/public/core']:
        if not (project/x).exists():raise SystemExit(f'项目缺少：{project/x}')
    old=summary(out,project)
    tasks=old['missing_triplets'][:args.max_triplets]
    print(f'已完成 {old["completed_triplets"]}/40，当前补跑 {len(tasks)} 组，workers={args.workers}',flush=True)
    def run_one(task):
        ds,seed=task['dataset'],task['seed']
        p=out/'by_seed'/ds/str(seed);p.mkdir(parents=True,exist_ok=True)
        log=p/'console.log'
        command=[sys.executable,str(script),'--project-root',str(project), '--datasets',ds,'--seeds',str(seed),
                 '--pop','30','--gens','192','--out',str(p)]
        # 单独子进程：正常完成后写原始 CSV；崩溃可用同一命令继续已有算法。
        with log.open('a',encoding='utf-8') as f:
            f.write('\n=== RESUME '+time.strftime('%Y-%m-%d %H:%M:%S')+' ===\n')
            f.flush()
            proc=subprocess.run(command,cwd=project,stdout=f,stderr=subprocess.STDOUT,check=False)
        valid=complete(p/'per_generation_hv.csv',ds,seed)
        return ds,seed,proc.returncode,valid
    failures=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_one,t):t for t in tasks}
        for future in as_completed(futures):
            try:
                ds,seed,rc,ok=future.result()
            except Exception as exc:
                failures.append({'task':futures[future],'error':str(exc),'traceback':traceback.format_exc()})
                print('失败',futures[future],str(exc),flush=True)
            else:
                if not ok:failures.append({'dataset':ds,'seed':seed,'exit_code':rc})
                done=summary(out,project)['completed_triplets']
                print(f'{ds} seed {seed}: exit={rc}, complete={ok}; 完成 {done}/40',flush=True)
    state=summary(out,project)
    (out/'failures.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'本轮结束：{state["completed_triplets"]}/40，失败={len(failures)}',flush=True)
    if failures:raise SystemExit(1)
if __name__=='__main__':main()
