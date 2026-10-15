# -*- coding: utf-8 -*-
# MSF-MOFS 正式复现源码。
# 中文阅读说明：本文件保留原版所有可执行语句与随机数调用顺序；本次仅增加中文注释。
# 修改优化器、目标函数或评估协议之前，必须重新进行完整论文结果回归。
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from .nsga2 import fast_non_dominated_sort, crowding_distance, hypervolume_2d

@dataclass
class SPEA2Result:
    population: np.ndarray
    objectives: np.ndarray
    fitness: np.ndarray
    history: list[dict]
    @property
    def pareto_mask(self):
        _, r = fast_non_dominated_sort(self.objectives)
        return r == 0
    @property
    def pareto_population(self): return self.population[self.pareto_mask]
    @property
    def pareto_objectives(self): return self.objectives[self.pareto_mask]

def _dominance_matrix(F):
    n=len(F); dom=np.zeros((n,n),bool)
    for i in range(n):
        for j in range(i+1,n):
            ij=bool(np.all(F[i]<=F[j]) and np.any(F[i]<F[j]))
            ji=bool(np.all(F[j]<=F[i]) and np.any(F[j]<F[i]))
            dom[i,j]=ij; dom[j,i]=ji
    return dom

# 计算 SPEA2 支配强度和密度适应度。
def _spea2_fitness(F):
    n=len(F)
    dom=_dominance_matrix(F)
    strength=dom.sum(axis=1).astype(float)
    raw=np.zeros(n,float)
    for i in range(n): raw[i]=strength[dom[:,i]].sum()
    # 在目标空间计算欧氏距离，k=floor(sqrt(n))，统计第k个非自身近邻。
    diff=F[:,None,:]-F[None,:,:]
    dist=np.sqrt(np.sum(diff*diff,axis=2))
    np.fill_diagonal(dist,np.inf)
    k=max(1,int(np.sqrt(n)))
    sigma=np.partition(dist,k-1,axis=1)[:,k-1]
    density=1.0/(sigma+2.0)
    return raw+density, strength, raw, density

def _truncate(indices, F, target):
    idx=list(map(int,indices))
    while len(idx)>target:
        V=F[idx]
        diff=V[:,None,:]-V[None,:,:]
        D=np.sqrt(np.sum(diff*diff,axis=2)); np.fill_diagonal(D,np.inf)
        # 按距离向量字典序比较拥挤程度，优先剔除最拥挤个体。
        sorted_d=np.sort(D,axis=1)
        remove=0
        for i in range(1,len(idx)):
            # 字典序更小的距离向量意味着周围更拥挤。
            a=sorted_d[i]; b=sorted_d[remove]
            for x,y in zip(a,b):
                if x < y - 1e-15: remove=i; break
                if x > y + 1e-15: break
        idx.pop(remove)
    return np.asarray(idx,dtype=int)

def _environmental_select(pop,F,archive_size):
    # 按稳定顺序去除重复染色体。
    _, ui=np.unique(pop,axis=0,return_index=True); ui=np.sort(ui); pop=pop[ui]; F=F[ui]
    fit,*_= _spea2_fitness(F)
    chosen=np.flatnonzero(fit < 1.0)
    if len(chosen) < archive_size:
        rest=np.setdiff1d(np.arange(len(F)),chosen,assume_unique=False)
        order=rest[np.argsort(fit[rest],kind='stable')]
        chosen=np.concatenate([chosen,order[:archive_size-len(chosen)]])
    elif len(chosen) > archive_size:
        chosen=_truncate(chosen,F,archive_size)
    cp=pop[chosen]; cF=F[chosen]
    cfit,*_=_spea2_fitness(cF)
    return cp,cF,cfit

def _ensure_nonempty(x,rng):
    if x.sum()==0: x[rng.integers(0,len(x))]=1
    return x

def _tournament(rng,fit):
    a,b=rng.integers(0,len(fit),size=2)
    if fit[a] < fit[b]: return int(a)
    if fit[b] < fit[a]: return int(b)
    return int(a if rng.random()<.5 else b)

def _offspring(archive,fitness,n,d,rng,cross,mut):
    children=[]
    while len(children)<n:
        p1=archive[_tournament(rng,fitness)]; p2=archive[_tournament(rng,fitness)]
        c1,c2=p1.copy(),p2.copy()
        if rng.random()<cross:
            mask=rng.random(d)<.5
            c1=np.where(mask,p1,p2).astype(np.uint8); c2=np.where(mask,p2,p1).astype(np.uint8)
        for c in (c1,c2):
            flip=rng.random(d)<mut; c[flip]=1-c[flip]; _ensure_nonempty(c,rng); children.append(c)
            if len(children)>=n: break
    return np.asarray(children,dtype=np.uint8)

# SPEA2 对照优化器，与 MSF 引导方案共享实验协议。
def run_spea2(evaluate: Callable[[np.ndarray],np.ndarray], n_features:int, population_size:int=40, generations:int=30,
              archive_size:int|None=None, crossover_prob:float=.9, mutation_prob:float|None=None,
              init_feature_prob:float=.2, random_state:int=42, verbose:bool=True, initial_population:np.ndarray|None=None):
    rng=np.random.default_rng(random_state); mutation_prob=(1.0/n_features) if mutation_prob is None else mutation_prob
    archive_size=population_size if archive_size is None else int(archive_size)
    if initial_population is None: pop=(rng.random((population_size,n_features))<init_feature_prob).astype(np.uint8)
    else:
        pop=np.asarray(initial_population,dtype=np.uint8).copy()
        if len(pop)>population_size: pop=pop[:population_size]
        elif len(pop)<population_size:
            extra=(rng.random((population_size-len(pop),n_features))<init_feature_prob).astype(np.uint8); pop=np.vstack([pop,extra])
    for x in pop: _ensure_nonempty(x,rng)
    F=np.vstack([evaluate(x) for x in pop])
    arch=np.empty((0,n_features),dtype=np.uint8); archF=np.empty((0,F.shape[1]),float); archfit=np.empty((0,),float)
    history=[]
    for gen in range(generations+1):
        if len(arch): combo=np.vstack([pop,arch]); comboF=np.vstack([F,archF])
        else: combo,comboF=pop,F
        arch,archF,archfit=_environmental_select(combo,comboF,archive_size)
        _,r=fast_non_dominated_sort(archF); pf=archF[r==0]
        owner=getattr(evaluate,'__self__',None); ec=len(getattr(owner,'cache',{})) if owner is not None else None
        history.append({'generation':gen,'pareto_size':int(len(pf)),'objective_min':pf.min(axis=0).tolist(),'objective_mean':pf.mean(axis=0).tolist(),'train_hypervolume_ref_1_1':hypervolume_2d(pf) if pf.shape[1]==2 else None,'unique_evaluations':ec,'archive_size':int(len(arch))})
        if gen==generations: break
        pop=_offspring(arch,archfit,population_size,n_features,rng,crossover_prob,mutation_prob)
        F=np.vstack([evaluate(x) for x in pop])
    return SPEA2Result(arch,archF,archfit,history)
