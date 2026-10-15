from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from irmofs.highdim_datasets import load_highdim_dataset
from irmofs.environments import make_nested_ir_environments
from irmofs.evaluator import FeatureSubsetEvaluator, heldout_gmeans
from irmofs.stable_guidance import fisher_scores, make_cisr_multiscale_population
from irmofs.highdim_formal import make_base_population
from irmofs.nsga2 import (
    NSGA2Result, fast_non_dominated_sort, crowding_distance, _ensure_nonempty,
    _offspring, _environmental_select, hypervolume_2d, run_nsga2,
)

DATASETS=['madelon','arcene']
SEEDS=list(range(72,82))
METHODS=['Random','Sparse-control','MSF-MOFS']
SNAP_GENS={0,1,3,6}


def mean_pairwise_normalized_hamming(pop: np.ndarray) -> float:
    P=np.asarray(pop,dtype=np.uint8)
    n,d=P.shape
    if n<2: return 0.0
    total=0.0; pairs=0
    for i in range(n-1):
        # uint8 inequality avoids constructing int differences
        total += np.not_equal(P[i+1:],P[i]).sum(axis=1).sum() / d
        pairs += n-i-1
    return float(total/pairs)


def run_nsga2_snapshots(evaluate,n_features,population_size,generations,random_state,initial_population):
    # Exact standard-NSGA-II path copied from the frozen run_nsga2 implementation,
    # adding read-only population snapshots after initialization/environmental selection.
    rng=np.random.default_rng(random_state)
    mutation_prob=1.0/n_features
    pop=np.asarray(initial_population,dtype=np.uint8).copy()
    if len(pop)>population_size: pop=pop[:population_size]
    elif len(pop)<population_size:
        extra=(rng.random((population_size-len(pop),n_features))<0.2).astype(np.uint8)
        pop=np.vstack([pop,extra])
    for i in range(population_size): _ensure_nonempty(pop[i],rng)
    F=np.vstack([evaluate(x) for x in pop])
    fronts,ranks=fast_non_dominated_sort(F); crowd=crowding_distance(F,fronts)
    snaps={0:pop.copy()}
    for gen in range(1,generations+1):
        while len(pop)<population_size:
            x=(rng.random(n_features)<0.2).astype(np.uint8); _ensure_nonempty(x,rng)
            pop=np.vstack([pop,x]); F=np.vstack([F,evaluate(x)])
        fronts,ranks=fast_non_dominated_sort(F); crowd=crowding_distance(F,fronts)
        child=_offspring(pop,ranks,crowd,rng,0.9,mutation_prob,mutation_strategy='standard',feature_frequency=None,alpha=0.0)
        child_F=np.vstack([evaluate(x) for x in child])
        pop,F,ranks,crowd=_environmental_select(np.vstack([pop,child]),np.vstack([F,child_F]),population_size)
        if gen in SNAP_GENS: snaps[gen]=pop.copy()
    return NSGA2Result(pop,F,ranks,crowd,[]),snaps


def front_test_hv(pareto_pop,train_envs,X_test,y_test,k=5):
    objs=[]
    for ch in pareto_pop:
        tg=heldout_gmeans(ch,train_envs,X_test,y_test,k)
        vals=np.asarray([tg[ir] for ir in sorted(tg)],float)
        objs.append([1.0-float(vals.mean()),float(np.mean(ch))])
    return float(hypervolume_2d(np.asarray(objs,float)))


def guided_candidates(d):
    vals=[]
    for x in [10,0.05*d,0.10*d,0.20*d]:
        k=max(1,min(d,int(round(x))))
        if k not in vals: vals.append(k)
    return vals


def write_csv(path, rows):
    if not rows: return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,default=Path('data'));ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)

    # 1) Fisher stability across outer seeds. Skip recomputation on resumed runs.
    pair_path=a.out/'fisher_stability_pairwise.csv'
    summary_path=a.out/'fisher_stability_summary.csv'
    if pair_path.exists() and summary_path.exists():
        print(json.dumps({'fisher_stability':'reuse_existing'},ensure_ascii=False),flush=True)
    else:
        pair_rows=[]
        for ds in DATASETS:
            bundle=load_highdim_dataset(ds,a.data_dir); d=bundle.X.shape[1]
            rank_by_seed={}; top_by_seed={}
            ks=guided_candidates(d)
            for seed in SEEDS:
                Xtr,_,ytr,_=train_test_split(bundle.X,bundle.y,test_size=.30,stratify=bundle.y,random_state=seed)
                sc=fisher_scores(Xtr,ytr)
                order=np.argsort(-sc,kind='stable')
                rv=np.empty(d,int);rv[order]=np.arange(1,d+1)
                rank_by_seed[seed]=rv
                top_by_seed[seed]={k:set(map(int,order[:k])) for k in ks}
            for i,s1 in enumerate(SEEDS):
                for s2 in SEEDS[i+1:]:
                    rho=float(spearmanr(rank_by_seed[s1],rank_by_seed[s2]).statistic)
                    pair_rows.append({'dataset':ds,'seed_a':s1,'seed_b':s2,'metric':'spearman_rank','k':'','value':rho,'k_fraction':''})
                    for k in ks:
                        A=top_by_seed[s1][k];B=top_by_seed[s2][k]
                        j=len(A&B)/len(A|B)
                        pair_rows.append({'dataset':ds,'seed_a':s1,'seed_b':s2,'metric':'topk_jaccard','k':k,'value':float(j),'k_fraction':k/d})
        write_csv(pair_path,pair_rows)
        summ=[]
        keys=sorted(set((r['dataset'],r['metric'],str(r['k'])) for r in pair_rows))
        for ds,metric,k in keys:
            x=np.array([r['value'] for r in pair_rows if r['dataset']==ds and r['metric']==metric and str(r['k'])==k],float)
            sample=next(r for r in pair_rows if r['dataset']==ds and r['metric']==metric and str(r['k'])==k)
            summ.append({'dataset':ds,'metric':metric,'k':k,'k_fraction':sample['k_fraction'],'n_pairs':len(x),'mean':float(x.mean()),'std':float(x.std(ddof=1)),'median':float(np.median(x)),'min':float(x.min()),'max':float(x.max())})
        write_csv(summary_path,summ)

    # 2) Diversity trajectories plus heldout PF-HV for context.
    raw_path=a.out/'population_diversity_raw.csv'
    existing=[]
    if raw_path.exists():
        with raw_path.open(newline='',encoding='utf-8') as f: existing=list(csv.DictReader(f))
    done={(r['dataset'],int(r['seed']),r['method']) for r in existing}
    runrows=existing
    for ds in DATASETS:
        bundle=load_highdim_dataset(ds,a.data_dir);D=bundle.X.shape[1]
        for seed in SEEDS:
            Xtr,Xte,ytr,yte=train_test_split(bundle.X,bundle.y,test_size=.30,stratify=bundle.y,random_state=seed)
            envs=make_nested_ir_environments(Xtr,ytr,[1,3,5],3,seed)
            scores=fisher_scores(Xtr,ytr)
            base=make_base_population(12,D,seed,.20)
            msf,sizes=make_cisr_multiscale_population(base,scores,guided_fraction=3/12)
            perm_scores=np.random.default_rng(seed+9001).permutation(scores)
            sparse,sizes2=make_cisr_multiscale_population(base,perm_scores,guided_fraction=3/12)
            assert sizes==sizes2 and len(sizes)==3
            init_map={'Random':base.copy(),'Sparse-control':sparse,'MSF-MOFS':msf}
            for method in METHODS:
                if (ds,seed,method) in done: continue
                ev=FeatureSubsetEvaluator(envs,mode='mean',n_neighbors=5,backend='sklearn')
                t0=time.perf_counter()
                res,snaps=run_nsga2_snapshots(ev.evaluate,D,12,6,seed,init_map[method])
                wall=time.perf_counter()-t0
                # Held-out PF-HV is not required for the P0-3 diversity diagnostic; skip it to avoid
                # confounding the diagnostic runtime with expensive high-dimensional test evaluation.
                hv=float('nan')
                for gen in sorted(SNAP_GENS):
                    p=snaps[gen]
                    runrows.append({
                      'dataset':ds,'seed':seed,'method':method,'generation':gen,
                      'mean_pairwise_normalized_hamming':mean_pairwise_normalized_hamming(p),
                      'population_size_snapshot':len(p),'mean_selected_fraction':float(p.mean()),
                      'train_hv_final':float(hypervolume_2d(res.pareto_objectives)),
                      'test_pf_hv_final':hv,'unique_wrapper_evaluations_final':len(ev.cache),
                      'guided_sizes':'/'.join(map(str,sizes)) if method!='Random' else '',
                      'search_wall_seconds':wall,
                    })
                write_csv(raw_path,runrows)
                done.add((ds,seed,method))
                print(json.dumps({'dataset':ds,'seed':seed,'method':method,'evals':len(ev.cache),'search_wall':wall},ensure_ascii=False),flush=True)

    # summaries
    dr=[]
    for ds in DATASETS:
      for method in METHODS:
       for gen in sorted(SNAP_GENS):
        rs=[r for r in runrows if r['dataset']==ds and r['method']==method and int(r['generation'])==gen]
        x=np.array([float(r['mean_pairwise_normalized_hamming']) for r in rs])
        dr.append({'dataset':ds,'method':method,'generation':gen,'n':len(x),'hamming_mean':float(x.mean()),'hamming_std':float(x.std(ddof=1)),'selected_fraction_mean':float(np.mean([float(r['mean_selected_fraction']) for r in rs]))})
    write_csv(a.out/'population_diversity_summary.csv',dr)
    protocol={'datasets':DATASETS,'seeds':SEEDS,'fisher_stability':'pairwise Spearman rank correlation plus Top-K Jaccard at K from {10,0.05D,0.10D,0.20D}','diversity':'mean pairwise normalized Hamming among current population at generations 0,1,3,6','methods':METHODS,'sparse_control':'matched base population; same three frozen MSF subset sizes; Fisher score vector permuted across feature identities with RNG seed+9001 before top-k seeding; standard NSGA-II thereafter','no_algorithm_tuning':True}
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')

if __name__=='__main__': main()
