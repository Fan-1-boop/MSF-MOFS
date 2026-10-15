#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从真实完整逐代记录画极简探索图，不补点，不为不足3种子的曲线画置信区间。"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t
METHODS=['MSF-MOFS','HFMOEA-CP','MBLPE-CP']
COLORS={'MSF-MOFS':'#245D83','HFMOEA-CP':'#B86D36','MBLPE-CP':'#28786F'}
STYLES={'MSF-MOFS':'-','HFMOEA-CP':'--','MBLPE-CP':':'}
LABELS={'hill_valley':'Hill-Valley','musk1':'MUSK1','madelon':'MADELON','arcene':'Arcene'}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.input,encoding='utf-8-sig')
    ds_order=[]
    matched={}
    for ds in ['madelon','hill_valley','musk1','arcene']:
        q=df[df.dataset.eq(ds)]
        counts=q.groupby(['seed','method']).generation.nunique().unstack(fill_value=0).reindex(columns=METHODS,fill_value=0)
        seeds=sorted(int(s) for s,r in counts.iterrows() if (r==193).all())
        if seeds:ds_order.append(ds);matched[ds]=seeds
    if not ds_order:raise SystemExit('暂无任一完整配对种子，无法画图')
    n=len(ds_order); ncols=2 if n>1 else 1; nrows=(n+ncols-1)//ncols
    fig,axs=plt.subplots(nrows,ncols,figsize=(7.2,3.1*nrows+0.4) if ncols==2 else (4.2,3.6),squeeze=False)
    for ax,ds in zip(axs.flat,ds_order):
        q=df[df.dataset.eq(ds)&df.seed.isin(matched[ds])]
        for method in METHODS:
            subset=q[q.method.eq(method)]
            s=subset.groupby('generation').test_pf_hv.agg(['mean','std','count']).reset_index()
            x=s.generation.to_numpy(float); y=s['mean'].to_numpy(float)
            ax.plot(x,y,color=COLORS[method],linestyle=STYLES[method],linewidth=1.55,label=method)
            if len(matched[ds])>=3:
                w=t.ppf(.975,s['count']-1)*s['std']/np.sqrt(s['count'])
                ax.fill_between(x,y-w,y+w,color=COLORS[method],alpha=.11,lw=0)
        ax.set_title(LABELS[ds],fontsize=10.2,pad=6)
        ax.grid(axis='y',alpha=.16,lw=.6)
        ax.set_xlim(0,192)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=8.2)
    for ax in list(axs.flat)[n:]:ax.axis('off')
    handles,labels=axs.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,fontsize=9,frameon=False,bbox_to_anchor=(.5,0.985))
    fig.supxlabel('Generation',y=.035,fontsize=10)
    fig.supylabel('Test HV',x=.02,fontsize=10)
    fig.subplots_adjust(left=.10,right=.99,top=.88,bottom=.14,hspace=.30,wspace=.22)
    stem=args.out/'Figure_pop30_g192_pilot'
    for ext in ['png','svg','pdf']:
        fig.savefig(stem.with_suffix('.'+ext),dpi=400,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    (args.out/'figure_metadata.json').write_text(json.dumps({'dataset_common_seeds':matched,'formal_4x10':all(len(matched.get(ds,[]))==10 for ds in LABELS)},ensure_ascii=False,indent=2),encoding='utf-8')
    print('已绘制',stem,'完整种子',matched)
if __name__=='__main__':main()
