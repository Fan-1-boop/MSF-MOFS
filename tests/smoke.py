#!/usr/bin/env python3
"""Lightweight non-benchmark verification: frozen MSF contract and MOEA API.
Only a deterministic toy problem; DOES NOT reproduce the 9-dataset results.
"""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from irmofs.stable_guidance import fisher_scores, make_cisr_multiscale_population
from irmofs.highdim_formal import make_base_population
from irmofs.nsga2 import run_nsga2,hypervolume_2d
from irmofs.spea2 import run_spea2

expected={100:[5,10,20],166:[8,10,17],500:[10,25,50],10000:[10,500,1000]}
rng=np.random.default_rng(123)
for d,ks in expected.items():
    x=rng.normal(size=(20,d))
    y=np.array([0]*10+[1]*10)
    score=fisher_scores(x,y)
    base=make_base_population(12,d,123,.20)
    guided,sizes=make_cisr_multiscale_population(base,score,guided_fraction=3/12)
    assert sizes==ks,(d,sizes,ks)
    assert np.array_equal(guided[3:],base[3:]), 'Non-guided individuals changed!'
    assert [int(v.sum()) for v in guided[:3]]==ks
    assert np.array_equal(base,make_base_population(12,d,123,.20)), 'Base population changed'
    print(f'PASS frozen seeding: D={d}, K={sizes}, other 9 individuals unchanged')

D=100
scores=np.linspace(1,0,D)
base=make_base_population(12,D,44,.20)
seed,_=make_cisr_multiscale_population(base,scores,guided_fraction=3/12)
def toy_evaluate(ch):
    z=np.asarray(ch,dtype=float)
    return np.array([1-float(z[:15].mean()),float(z.mean())])
for carrier,runner in [('NSGA-II',run_nsga2),('SPEA2',run_spea2)]:
    kw=dict(evaluate=toy_evaluate,n_features=D,population_size=12,generations=1,random_state=44,initial_population=seed)
    if carrier=='SPEA2': kw['archive_size']=12
    result=runner(**kw)
    hv=hypervolume_2d(result.pareto_objectives)
    assert 0<=hv<=1, (carrier,hv)
    print(f'PASS {carrier}: toy one-generation search, HV={hv:.6f}')
print('SMOKE TEST PASSED: not a benchmark reproduction.')
