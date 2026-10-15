
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, os, platform, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import __version__ as sklearn_version
# 与操作系统无关的项目定位；允许从任意目录调用分析脚本。
pre=argparse.ArgumentParser(add_help=False)
pre.add_argument('--project-root', type=Path, default=Path.cwd())
ROOT=pre.parse_known_args()[0].project_root.resolve()
if not (ROOT/'src/irmofs').is_dir():
    raise SystemExit(f'找不到项目核心源码：{ROOT}/src/irmofs，请传入 --project-root')
sys.path.insert(0, str(ROOT/'src'))
from irmofs.highdim_datasets import load_highdim_dataset
from irmofs.environments import make_nested_ir_environments
from irmofs.evaluator import FeatureSubsetEvaluator, heldout_gmeans
from irmofs.stable_guidance import fisher_scores, make_cisr_multiscale_population
from irmofs.highdim_formal import make_base_population
from irmofs.nsga2 import fast_non_dominated_sort, crowding_distance, _ensure_nonempty, _offspring, _environmental_select, hypervolume_2d
from irmofs.external_baselines import _hfmoea_filter_rankings, _mblpe_environmental_select, _cal_mblpe_fitness
METHODS=['MSF-MOFS','HFMOEA-CP','MBLPE-CP']
DATASETS=['madelon','hill_valley','musk1','arcene']

def pareto_idx(F):
    _, rank = fast_non_dominated_sort(np.asarray(F, float))
    return np.flatnonzero(rank==0)

def front_test_hv(pop, Ftrain, envs, Xte, yte, test_cache):
    idx = pareto_idx(Ftrain)
    objs=[]
    for i in idx:
        mask=np.asarray(pop[int(i)], np.uint8)
        key=mask.tobytes()
        if key not in test_cache:
            scores=heldout_gmeans(mask, envs, Xte, yte, 5)
            test_cache[key]=float(np.mean(list(scores.values())))
        objs.append([1.0-test_cache[key], float(mask.mean())])
    return float(hypervolume_2d(np.asarray(objs,float))), int(len(idx))

def run_msf_snapshots(Xtr,ytr,envs,D,pop_size,gens,seed):
    evaluator=FeatureSubsetEvaluator(envs, mode='mean', n_neighbors=5, backend='sklearn')
    rng=np.random.default_rng(seed)
    scores=fisher_scores(Xtr,ytr)
    base=make_base_population(pop_size,D,seed,0.20)
    init,_=make_cisr_multiscale_population(base,scores,guided_fraction=3.0/pop_size)
    pop=np.asarray(init,dtype=np.uint8).copy()
    if len(pop)>pop_size: pop=pop[:pop_size]
    elif len(pop)<pop_size:
        extra=(rng.random((pop_size-len(pop),D))<0.2).astype(np.uint8)
        pop=np.vstack([pop,extra])
    for i in range(pop_size): _ensure_nonempty(pop[i],rng)
    F=np.vstack([evaluator.evaluate(x) for x in pop])
    snaps=[(0,pop.copy(),F.copy(),len(evaluator.cache))]
    mutation_prob=1.0/D
    for gen in range(1,gens+1):
        fronts,ranks=fast_non_dominated_sort(F); crowd=crowding_distance(F,fronts)
        child=_offspring(pop,ranks,crowd,rng,0.9,mutation_prob,mutation_strategy='standard',feature_frequency=None,alpha=0.0)
        child_F=np.vstack([evaluator.evaluate(x) for x in child])
        pop,F,_,_=_environmental_select(np.vstack([pop,child]),np.vstack([F,child_F]),pop_size)
        snaps.append((gen,pop.copy(),F.copy(),len(evaluator.cache)))
    return snaps, evaluator

def run_hf_snapshots(Xtr,ytr,envs,D,pop_size,gens,seed):
    evaluator=FeatureSubsetEvaluator(envs, mode='mean', n_neighbors=5, backend='sklearn')
    rng=np.random.default_rng(seed)
    rankings=_hfmoea_filter_rankings(Xtr,ytr,seed)
    pop=np.zeros((pop_size,D),dtype=np.uint8)
    k=min(max(1,10),D)
    for i, ranks in enumerate(rankings[:10]):
        pop[i, np.flatnonzero(ranks <= k)] = 1
    if pop_size>10:
        pop[10:]=rng.integers(0,2,size=(pop_size-10,D),dtype=np.uint8)
    for i in range(pop_size): pop[i]=_ensure_nonempty(pop[i],rng)
    F=np.vstack([evaluator.evaluate(x) for x in pop])
    snaps=[(0,pop.copy(),F.copy(),len(evaluator.cache))]
    midpoint=D//2
    mutation_percent=6.0
    for gen in range(1, gens+1):
        children=np.empty_like(pop)
        for i in range(pop_size):
            p1=pop[i % pop_size]; p2=pop[(i+1)%pop_size]
            children[i,:midpoint]=p1[:midpoint]
            children[i,midpoint:]=p2[midpoint:]
        num_mut=max(1,int(pop_size*D*mutation_percent/100.0))
        mut_idx=np.unique(rng.integers(0,D,size=num_mut))
        children[:,mut_idx]=1-children[:,mut_idx]
        for i in range(pop_size): children[i]=_ensure_nonempty(children[i],rng)
        CF=np.vstack([evaluator.evaluate(x) for x in children])
        pop,F,_,_=_environmental_select(np.vstack([pop,children]),np.vstack([F,CF]),pop_size)
        snaps.append((gen,pop.copy(),F.copy(),len(evaluator.cache)))
    return snaps, evaluator

def run_mblpe_snapshots(Xtr,ytr,envs,D,pop_size,gens,seed):
    evaluator=FeatureSubsetEvaluator(envs, mode='mean', n_neighbors=5, backend='sklearn')
    rng=np.random.default_rng(seed)
    T=min(D,pop_size*3)
    scores=fisher_scores(Xtr,ytr)
    selected=np.argsort(-scores,kind='stable')[:T]
    d=len(selected)
    threshold=0.6
    cache_red={}
    def eval_reduced(z):
        z=_ensure_nonempty(np.asarray(z,np.uint8),rng)
        key=z.tobytes()
        if key in cache_red: return cache_red[key].copy()
        full=np.zeros(D,dtype=np.uint8); full[selected]=z
        val=np.asarray(evaluator.evaluate(full),float).copy()
        val[1]=float(z.mean())
        cache_red[key]=val
        return val.copy()
    def expand_pop(red_pop):
        full=np.zeros((len(red_pop),D),dtype=np.uint8)
        full[:,selected]=red_pop
        return full
    def common_F(red_pop, reduced_F):
        # 只恢复第二目标的原始D维比例；第一目标不变，避免旁路评价改变缓存计数。
        out=np.asarray(reduced_F,float).copy()
        out[:,1]=np.asarray(red_pop,np.uint8).sum(axis=1)/float(D)
        return out
    Fscale, CR = 0.7, 0.7
    # MBLPE 原版把初始化算作第1代；为了横轴统一为“初始化后演化轮次”，
    # 额外增加1代原版预算，使三种方法都拥有 gen=0..gens 个真实状态。
    max_fes = pop_size * (gens + 1)
    pop=(rng.random((pop_size,d))>threshold).astype(np.uint8)
    for i in range(pop_size): pop[i]=_ensure_nonempty(pop[i],rng)
    F=np.vstack([eval_reduced(x) for x in pop])
    fitcount=pop_size
    chain=[(pop.copy(),F.copy())]
    snaps=[(0, expand_pop(pop), common_F(pop,F),len(evaluator.cache))]
    gen=1
    # MBLPE 初始代在原实现中记为 generation=1；此处统一将初始化记为0。
    # 因此运行 gens 次实际演化，等价于原函数 generations=gens+1。
    while gen<=gens:
        if gen in (1,2):  # 原版 generation=2,3 对应统一时间轴的第 1、2 次演化
            trials=[]
            for i in range(pop_size):
                candidates=np.delete(np.arange(pop_size),i)
                a,b,c=rng.choice(candidates,size=3,replace=False)
                mutant=pop[a].astype(float)+Fscale*(pop[b].astype(float)-pop[c].astype(float))
                bad=(mutant<0)|(mutant>1)
                mutant[bad]=rng.random(bad.sum())
                trial=pop[i].astype(float).copy()
                jrand=rng.integers(0,d)
                mask=rng.random(d)<=CR
                mask[jrand]=True
                trial[mask]=mutant[mask]
                trial=(trial>threshold).astype(np.uint8)
                trials.append(_ensure_nonempty(trial,rng))
            trials=np.unique(np.asarray(trials,np.uint8),axis=0)
            TF=np.vstack([eval_reduced(x) for x in trials])
            fitcount += len(trials)
            pop,F=_mblpe_environmental_select(np.vstack([trials,pop]),np.vstack([TF,F]),pop_size)
            chain.append((pop.copy(),F.copy()))
        else:
            if fitcount >= max_fes:
                # 原版因评价预算终止；不凭空延伸水平线来假装算法已收敛。
                break
            p1,p2,p3=chain[-3][0],chain[-2][0],chain[-1][0]
            i1,i2,i3=rng.permutation(pop_size),rng.permutation(pop_size),rng.permutation(pop_size)
            trial_cont=(4.0*p3[i3].astype(float)+p2[i2].astype(float)-2.0*p1[i1].astype(float))/3.0
            bad=(trial_cont<0)|(trial_cont>1)
            trial_cont[bad]=rng.random(bad.sum())
            trials=(trial_cont>threshold).astype(np.uint8)
            for i in range(len(trials)): trials[i]=_ensure_nonempty(trials[i],rng)
            trials=np.unique(trials,axis=0)
            seen={x.tobytes() for p,_ in chain[-3:] for x in p}
            filtered=[x for x in trials if x.tobytes() not in seen]
            trials=np.asarray(filtered,dtype=np.uint8) if filtered else np.empty((0,d),dtype=np.uint8)
            if len(trials)==0:
                trials=(rng.random((pop_size,d))>threshold).astype(np.uint8)
                for i in range(len(trials)): trials[i]=_ensure_nonempty(trials[i],rng)
                trials=np.unique(trials,axis=0)
            remaining=max_fes-fitcount
            trials=trials[:remaining]
            if len(trials)>0:
                TF=np.vstack([eval_reduced(x) for x in trials])
                fitcount += len(trials)
                pop,F=_mblpe_environmental_select(np.vstack([trials,chain[-1][0]]),np.vstack([TF,chain[-1][1]]),pop_size)
                chain.append((pop.copy(),F.copy())); chain=chain[-3:]
        snaps.append((gen, expand_pop(pop), common_F(pop,F),len(evaluator.cache)))
        gen += 1
    return snaps, evaluator

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project-root', type=Path, default=ROOT)
    ap.add_argument('--datasets', nargs='+', default=DATASETS)
    ap.add_argument('--seeds', nargs='+', type=int, default=list(range(92,102)))
    ap.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    ap.add_argument('--gens', type=int, default=96)
    ap.add_argument('--pop', type=int, default=12)
    ap.add_argument('--data-dir', type=Path, default=ROOT/'data/public/core')
    ap.add_argument('--out', type=Path, required=True)
    args=ap.parse_args()
    if args.gens<3: raise SystemExit('为保留MBLPE的三阶段机制，--gens不能小于3')
    args.out.mkdir(parents=True, exist_ok=True)
    protocol_path=args.out/'protocol.json'
    if protocol_path.exists():
        previous=json.loads(protocol_path.read_text(encoding='utf-8'))
        expected={'datasets':args.datasets,'seeds':args.seeds,'methods':args.methods,'generations':args.gens,'population':args.pop}
        if any(previous.get(k)!=v for k,v in expected.items()):
            raise SystemExit('当前输出目录已有其他实验协议，请使用新的 --out，避免混合实验')
    # 支持中断后继续，避免耗时的高维 HFMOEA 排序被重复计算。
    raw=[]
    output_csv=args.out/'per_generation_hv.csv'
    if output_csv.exists():
        existing=pd.read_csv(output_csv,encoding='utf-8-sig')
        if {'dataset','seed','method','generation'}.issubset(existing.columns):
            raw=existing.to_dict('records')
    protocol={'experiment':'generation-convergence-test-HV','datasets':args.datasets,'seeds':args.seeds,'methods':args.methods,'generations':args.gens,
              'population':args.pop,'mblpe_generation_convention':'初始状态记为0；MBLPE 原函数 generations=统一演化轮次+1','x_axis':'generation','y_axis':'held-out pareto-front hypervolume','python':sys.version,
              'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn_version,'os':platform.platform(),'cpu_count':os.cpu_count()}
    (args.out/'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    for ds in args.datasets:
        bundle=load_highdim_dataset(ds,args.data_dir)
        D=bundle.X.shape[1]
        for seed in args.seeds:
            print(f'=== {ds} seed={seed} ===', flush=True)
            Xtr,Xte,ytr,yte=train_test_split(bundle.X,bundle.y,test_size=.30,stratify=bundle.y,random_state=seed)
            envs=make_nested_ir_environments(Xtr,ytr,[1,3,5],3,seed)
            for method in args.methods:
                expected_end=args.gens
                existing_gens={int(r['generation']) for r in raw if r['dataset']==ds and int(r['seed'])==seed and r['method']==method}
                if existing_gens==set(range(expected_end+1)):
                    print(f'  {method}: 已完成，跳过',flush=True)
                    continue
                if existing_gens:
                    raw=[r for r in raw if not (r['dataset']==ds and int(r['seed'])==seed and r['method']==method)]
                t0=time.perf_counter()
                if method=='MSF-MOFS':
                    snaps, evaluator = run_msf_snapshots(Xtr,ytr,envs,D,args.pop,args.gens,seed)
                elif method=='HFMOEA-CP':
                    snaps, evaluator = run_hf_snapshots(Xtr,ytr,envs,D,args.pop,args.gens,seed)
                else:
                    snaps, evaluator = run_mblpe_snapshots(Xtr,ytr,envs,D,args.pop,args.gens,seed)
                test_cache={}
                for gen,pop,Ftr,eval_count in snaps:
                    hv,pf_size=front_test_hv(pop,Ftr,envs,Xte,yte,test_cache)
                    raw.append({'dataset':ds,'seed':seed,'method':method,'generation':gen,'test_pf_hv':hv,'train_pf_hv':float(hypervolume_2d(np.asarray(Ftr[pareto_idx(Ftr)],float))),
                                'pf_size':pf_size,'unique_wrapper_evaluations':int(eval_count)})
                wall=time.perf_counter()-t0
                print(f'  {method}: done {len(snaps)} gens; unique={len(evaluator.cache)}; wall={wall:.1f}s', flush=True)
                # 原子写入：防止任务中断留下半行或损坏的数据文件。
                tmp=output_csv.with_suffix('.csv.tmp')
                pd.DataFrame(raw).to_csv(tmp, index=False, encoding='utf-8-sig')
                tmp.replace(output_csv)
    df=pd.DataFrame(raw)
    summary=df.groupby(['dataset','generation','method'], as_index=False).test_pf_hv.agg(mean='mean', std='std', n='count')
    summary.to_csv(args.out/'generation_hv_summary.csv', index=False, encoding='utf-8-sig')
    print('completed', args.out, flush=True)

if __name__=='__main__':
    main()
