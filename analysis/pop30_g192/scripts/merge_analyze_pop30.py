#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合并 Population=30/192代的真实逐代记录；只收录完整 0..192 轮的实验。"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

METHODS=['MSF-MOFS','HFMOEA-CP','MBLPE-CP']
DATASETS=['madelon','hill_valley','musk1','arcene']
COLS=['dataset','seed','method','generation','test_pf_hv','train_pf_hv','pf_size','unique_wrapper_evaluations']
CHECKS=[0,6,24,48,96,144,192]

def merge(inputs:list[Path]):
    all_rows=[]; rejected=[]
    for fp in inputs:
        if not fp.exists():
            rejected.append({'path':str(fp),'reason':'文件不存在'});continue
        df=pd.read_csv(fp,encoding='utf-8-sig')
        if set(COLS)-set(df.columns):
            rejected.append({'path':str(fp),'reason':'字段不完整'});continue
        for (ds,seed,method),part in df.groupby(['dataset','seed','method']):
            part=part.sort_values('generation')
            gen=part.generation.to_numpy(int)
            ident=f'{ds}/{int(seed)}/{method}'
            if not np.array_equal(gen,np.arange(193)):
                rejected.append({'path':str(fp),'group':ident,'reason':f'代数不完整（共{len(gen)}行）'});continue
            if not np.isfinite(part[['train_pf_hv','test_pf_hv']].to_numpy(float)).all():
                rejected.append({'path':str(fp),'group':ident,'reason':'HV有无效数字'});continue
            vals=part.unique_wrapper_evaluations.to_numpy(int)
            if (np.diff(vals)<0).any():
                rejected.append({'path':str(fp),'group':ident,'reason':'唯一评价次数下降'});continue
            part=part[COLS].copy()
            part['source_file']=str(fp)
            all_rows.append(part)
    if not all_rows:
        raise SystemExit('未发现任何完整记录')
    data=pd.concat(all_rows,ignore_index=True)
    groups=['dataset','seed','method','generation']
    if data.duplicated(groups).any():
        duplicated=data[data.duplicated(groups,keep=False)].sort_values(groups)
        for _, grp in duplicated.groupby(groups):
            if not np.allclose(grp[['test_pf_hv','train_pf_hv']].to_numpy(float),grp[['test_pf_hv','train_pf_hv']].iloc[0].to_numpy(float), rtol=0,atol=1e-12):
                raise ValueError('同一记录出现不一致的重复数值：'+str(grp[groups].iloc[0].to_dict()))
        data=data.drop_duplicates(groups,keep='first')
    return data.sort_values(groups).reset_index(drop=True),rejected

def compute(df):
    rows=[]
    for (ds,seed,method), grp in df.groupby(['dataset','seed','method']):
        g=grp.set_index('generation')
        last=g.loc[192]
        old=g.loc[168]
        best=np.maximum.accumulate(g.train_pf_hv.to_numpy(float))
        delta=float(best[-1]-best[-25])
        rec={'dataset':ds,'seed':int(seed),'method':method,'population':30,'max_generation':192,
             'test_hv_g0':float(g.loc[0,'test_pf_hv']),
             'test_hv_g6':float(g.loc[6,'test_pf_hv']),
             'test_hv_g48':float(g.loc[48,'test_pf_hv']),
             'test_hv_g96':float(g.loc[96,'test_pf_hv']),
             'test_hv_g144':float(g.loc[144,'test_pf_hv']),
             'test_hv_g192':float(last.test_pf_hv),
             'train_hv_g192':float(last.train_pf_hv),
             'train_best_gain_g168_to_g192':delta,
             'empirical_train_plateau':bool(delta<=0.005),
             'unique_wrapper_evals_g192':int(last.unique_wrapper_evaluations)}
        rows.append(rec)
    run=pd.DataFrame(rows).sort_values(['dataset','seed','method'])
    checkpoints=df[df.generation.isin(CHECKS)].copy()
    summary=checkpoints.groupby(['dataset','method','generation'],as_index=False).agg(
        mean_test_hv=('test_pf_hv','mean'),sd_test_hv=('test_pf_hv','std'),n_runs=('seed','nunique'))
    # 只对相同seed三算法均具备完整结果的情况给出胜负计数，避免部分种子产生偏差。
    pairs=[]
    for ds,grp in run.groupby('dataset'):
        piv=grp.pivot(index='seed',columns='method',values='test_hv_g192').reindex(columns=METHODS).dropna(subset=METHODS)
        for opponent in METHODS[1:]:
            delta=piv['MSF-MOFS']-piv[opponent]
            pairs.append({'dataset':ds,'opponent':opponent,'n_matched_seeds':len(delta),
                 'msf_wins':int((delta>1e-12).sum()),'ties':int((delta.abs()<=1e-12).sum()),
                 'msf_losses':int((delta<-1e-12).sum()),
                 'mean_hv_diff_msf_minus_opponent':float(delta.mean()) if len(delta) else None})
    return run,summary,pd.DataFrame(pairs)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',nargs='+',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference-pop12',type=Path,default=None,help='已核验的Pop12逐代CSV（只用于参考差值）')
    p.add_argument('--require-complete', action='store_true', help='强制要求四数据集十种子三方法共120次完整搜索')
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    df, rejected=merge(a.inputs)
    run,summary,pairs=compute(df)
    expected={(ds,seed,method) for ds in DATASETS for seed in range(92,102) for method in METHODS}
    actual={(str(row.dataset),int(row.seed),str(row.method)) for row in run.itertuples()}
    if a.require_complete and actual!=expected:
        raise SystemExit(f'正式版完整性未通过：已完成{len(actual & expected)}/{len(expected)}；缺少{len(expected-actual)}次，拒绝生成正式统计')
    df.drop(columns=['source_file']).to_csv(a.out/'per_generation_hv.csv',index=False,encoding='utf-8-sig')
    run.to_csv(a.out/'completed_run_endpoints.csv',index=False,encoding='utf-8-sig')
    summary.to_csv(a.out/'selected_generation_summary.csv',index=False,encoding='utf-8-sig')
    pairs.to_csv(a.out/'paired_comparison_g192.csv',index=False,encoding='utf-8-sig')
    if a.reference_pop12 is not None:
        old=pd.read_csv(a.reference_pop12,encoding='utf-8-sig')
        old=old[old.generation.eq(96)][['dataset','seed','method','test_pf_hv','unique_wrapper_evaluations']]
        old=old.rename(columns={'test_pf_hv':'test_hv_pop12_g96','unique_wrapper_evaluations':'unique_evals_pop12_g96'})
        compared=run.merge(old,on=['dataset','seed','method'],how='inner')
        compared['delta_hv_pop30_minus_pop12_g96']=compared['test_hv_g96']-compared['test_hv_pop12_g96']
        compared.to_csv(a.out/'pop12_vs_pop30_at_g96.csv',index=False,encoding='utf-8-sig')
    meta={
       'experiment_type':'探索性补充实验，不是冻结Table2',
       'population':30,'generation_limit':192,'seeds_expected':list(range(92,102)),
       'completed_algorithms':len(run),'completed_triplets':int(sum(len(g)==3 for _,g in run.groupby(['dataset','seed']))),
       'total_target_algorithms':120,'n_dataset_seeds_target':40,
       'plateaued_runs':int(run.empirical_train_plateau.sum()),
       'plateau_metric':'训练HV历史最佳值在最后24轮的增量 <=0.005',
       'rejected':rejected,
       'per_dataset':{ds:{'completed_algorithms':int(len(run[run.dataset==ds])),
                         'seeds_with_all_three':sorted(int(sd) for sd,g in run[run.dataset==ds].groupby('seed') if len(g)==3)} for ds in DATASETS}
    }
    (a.out/'verification.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(meta,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
