from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import wilcoxon, rankdata
from sklearn.model_selection import train_test_split

from irmofs.datasets import DatasetBundle, _clean_xy
from irmofs.environments import make_nested_ir_environments
from irmofs.evaluator import FeatureSubsetEvaluator, heldout_gmeans
from irmofs.stable_guidance import fisher_scores, make_cisr_multiscale_population
from irmofs.highdim_formal import make_base_population
from irmofs.nsga2 import run_nsga2, hypervolume_2d
from irmofs.spea2 import run_spea2

LOCKED = ['ALLAML','colon','GLI-85','Prostate-GE','SMK-CAN-187']
CARRIERS = ['NSGA-II','SPEA2']
INITS = ['Random','MSF']


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()


def load_locked_dataset(name: str, data_dir: Path) -> tuple[DatasetBundle, Path]:
    key=name.lower()
    if key=='colon':
        p=data_dir/'pone.0246039.s002.csv'
        df=pd.read_csv(p)
        if 'response' not in df.columns: raise ValueError('colon CSV missing response')
        feature_cols=[c for c in df.columns if c.startswith('V')]
        if len(feature_cols)!=2000: raise ValueError(f'colon expected 2000 V features, got {len(feature_cols)}')
        X,y,names=_clean_xy(df[feature_cols],df['response'])
        return DatasetBundle('colon',X,y,names),p
    filemap={
      'allaml':'ALLAML.mat','gli-85':'GLI-85.mat','prostate-ge':'Prostate_GE.mat','smk-can-187':'SMK_CAN_187.mat'
    }
    if key not in filemap: raise ValueError(name)
    p=data_dir/filemap[key]
    m=sio.loadmat(p)
    if 'X' not in m or 'Y' not in m: raise ValueError(f'{p} missing X/Y')
    X,y,names=_clean_xy(m['X'],m['Y'])
    return DatasetBundle(name,X,y,names),p


def front_test_metrics(pareto_pop, train_envs, X_test, y_test, k=5):
    objs=[]; means=[]; worsts=[]; counts=[]
    for c in pareto_pop:
        tg=heldout_gmeans(c,train_envs,X_test,y_test,k)
        vals=np.asarray([tg[ir] for ir in sorted(tg)],float)
        mg=float(vals.mean()); means.append(mg); worsts.append(float(vals.min())); counts.append(int(c.sum()))
        objs.append([1.0-mg,float(np.mean(c))])
    F=np.asarray(objs,float)
    return dict(test_pf_hv=float(hypervolume_2d(F)),pf_size=int(len(pareto_pop)),
                best_test_mean_gmean_on_pf=float(np.max(means)),best_test_worst_gmean_on_pf=float(np.max(worsts)),
                min_features_on_pf=int(np.min(counts)))


def rank_biserial(diff):
    d=np.asarray(diff,float); nz=d[~np.isclose(d,0,atol=1e-15,rtol=0)]
    if len(nz)==0:return 0.0
    r=rankdata(np.abs(nz),method='average'); wp=r[nz>0].sum(); wm=r[nz<0].sum()
    return float((wp-wm)/(wp+wm)) if wp+wm else 0.0


def holm(pvals):
    p=np.asarray(pvals,float); m=len(p); order=np.argsort(p); adj=np.empty(m); running=0.0
    for rank,idx in enumerate(order):
        running=max(running,(m-rank)*p[idx]); adj[idx]=min(1.0,running)
    return adj


def write_stats(raw:Path,out:Path):
    df=pd.read_csv(raw)
    df.groupby(['dataset','carrier','init'],as_index=False).agg(
      n=('seed','count'),test_pf_hv_mean=('test_pf_hv','mean'),test_pf_hv_std=('test_pf_hv','std'),
      train_hv_mean=('train_hv','mean'),train_hv_std=('train_hv','std'),
      unique_evals_mean=('unique_wrapper_evaluations','mean'),search_time_mean=('search_time','mean'),
      heldout_time_mean=('heldout_evaluation_time','mean'),total_time_mean=('total_time','mean')
    ).to_csv(out/'summary.csv',index=False)
    rows=[]
    for (carrier,ds),g in df.groupby(['carrier','dataset']):
        a=g[g['init']=='MSF'].set_index('seed')['test_pf_hv']; b=g[g['init']=='Random'].set_index('seed')['test_pf_hv']
        idx=a.index.intersection(b.index); diff=(a.loc[idx]-b.loc[idx]).to_numpy(float)
        if not len(diff):continue
        try:stat,p=wilcoxon(diff,zero_method='wilcox',alternative='two-sided',method='auto')
        except ValueError:stat,p=0.,1.
        rows.append(dict(dataset=ds,carrier=carrier,n=len(diff),mean_difference=float(diff.mean()),median_difference=float(np.median(diff)),
                         wins=int((diff>1e-15).sum()),ties=int((np.abs(diff)<=1e-15).sum()),losses=int((diff<-1e-15).sum()),
                         rank_biserial=rank_biserial(diff),wilcoxon_stat=float(stat),p_raw=float(p)))
    pr=pd.DataFrame(rows)
    if len(pr):
        pr['p_holm_within_carrier']=np.nan
        for carrier,idx in pr.groupby('carrier').groups.items(): pr.loc[idx,'p_holm_within_carrier']=holm(pr.loc[idx,'p_raw'].values)
    pr.to_csv(out/'paired_msf_vs_random.csv',index=False)
    # carrier interaction: paired difference of MSF-Random gains across carriers
    ints=[]
    for ds,g in df.groupby('dataset'):
        piv=g.pivot_table(index='seed',columns=['carrier','init'],values='test_pf_hv',aggfunc='first')
        need=[('NSGA-II','MSF'),('NSGA-II','Random'),('SPEA2','MSF'),('SPEA2','Random')]
        if not all(c in piv.columns for c in need): continue
        d=((piv[('NSGA-II','MSF')]-piv[('NSGA-II','Random')])-(piv[('SPEA2','MSF')]-piv[('SPEA2','Random')])).dropna().values
        if not len(d):continue
        try:stat,p=wilcoxon(d,zero_method='wilcox',alternative='two-sided',method='auto')
        except ValueError:stat,p=0.,1.
        ints.append(dict(dataset=ds,n=len(d),mean_gain_difference_nsga_minus_spea=float(np.mean(d)),wins=int((d>1e-15).sum()),ties=int((np.abs(d)<=1e-15).sum()),losses=int((d<-1e-15).sum()),wilcoxon_stat=float(stat),p_raw=float(p)))
    it=pd.DataFrame(ints)
    if len(it):it['p_holm_across_datasets']=holm(it['p_raw'].values)
    it.to_csv(out/'carrier_interaction.csv',index=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--datasets',nargs='+',default=LOCKED);ap.add_argument('--seeds',nargs='+',type=int,default=list(range(92,102)))
    ap.add_argument('--data-dir',type=Path,default=Path('/mnt/data'));ap.add_argument('--out',type=Path,required=True);ap.add_argument('--pop',type=int,default=12);ap.add_argument('--gen',type=int,default=6)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);raw=a.out/'raw.csv'
    # Materialize and validate all named datasets before writing protocol.
    manifests=[]
    expected={'ALLAML':(72,7129),'colon':(62,2000),'GLI-85':(85,22283),'Prostate-GE':(102,5966),'SMK-CAN-187':(187,19993)}
    bundles={}
    for ds in a.datasets:
        b,p=load_locked_dataset(ds,a.data_dir);bundles[ds]=b
        if ds in expected and b.X.shape!=expected[ds]:raise RuntimeError(f'{ds} shape {b.X.shape} != locked {expected[ds]}')
        manifests.append(dict(dataset=ds,path=str(p),sha256=sha256(p),n_samples=int(b.X.shape[0]),n_features=int(b.X.shape[1]),class_counts={str(int(k)):int(v) for k,v in zip(*np.unique(b.y,return_counts=True))}))
    proto={
      'status':'LOCKED_BEFORE_UNTOUCHED_BREADTH_RESULTS','purpose':'untouched-dataset breadth validation of the frozen mixed_m3 initializer across two binary MOEA carriers',
      'selection_rule':'ASU Biological Data; binary; D>=2000; 50<=N<=200; exclude duplicate leukemia representation; all five locked datasets must be reported without filtering',
      'datasets':a.datasets,'dataset_manifests':manifests,'seeds':a.seeds,'population':a.pop,'generations':a.gen,
      'carriers':CARRIERS,'comparison':'MSF vs matched Random within each carrier/dataset',
      'msf':'one full outer-training Fisher ranking; exactly 3 guided chromosomes; smallest three distinct integer sizes from {10,0.05D,0.10D,0.20D}; standard downstream carrier thereafter',
      'matched_initialization':'same deterministic random base population; MSF replaces exactly first 3 chromosomes',
      'outer_split':'70/30 stratified, random_state=seed','wrapper':'same 5-NN / 3-fold / mean G-mean across IR={1,3,5}; feature-ratio second objective',
      'primary_endpoint':'held-out Pareto-set HV','secondary_endpoints':['train HV','best held-out mean G-mean on PF','best held-out worst-IR G-mean on PF','minimum PF feature count','unique wrapper evaluations','wall-clock decomposition'],
      'primary_inference':'paired two-sided Wilcoxon MSF vs Random, Holm corrected across 5 datasets separately within each carrier; rank-biserial effect size; wins/ties/losses',
      'carrier_interaction':'paired difference in MSF-Random gain between NSGA-II and SPEA2, Holm corrected across 5 datasets',
      'no_result_filtering':True,'method_selection':'none; these untouched datasets cannot change mixed_m3 or its schedule'
    }
    (a.out/'PROTOCOL_BEFORE_RESULTS.json').write_text(json.dumps(proto,indent=2),encoding='utf-8')
    rows=pd.read_csv(raw).to_dict('records') if raw.exists() else [];done={(r['dataset'],int(r['seed']),r['carrier'],r['init']) for r in rows}
    for ds in a.datasets:
      bundle=bundles[ds];D=bundle.X.shape[1]
      for seed in a.seeds:
        Xtr,Xte,ytr,yte=train_test_split(bundle.X,bundle.y,test_size=.30,stratify=bundle.y,random_state=seed)
        envs=make_nested_ir_environments(Xtr,ytr,[1,3,5],3,seed)
        t_rank=time.perf_counter();scores=fisher_scores(Xtr,ytr);ranking_time=time.perf_counter()-t_rank
        base=make_base_population(a.pop,D,seed,.20);msf,sizes=make_cisr_multiscale_population(base,scores,guided_fraction=3/a.pop)
        if len(sizes)!=3:raise RuntimeError(f'{ds} {seed} sizes={sizes}')
        for carrier in CARRIERS:
          for init_name,init in [('Random',base),('MSF',msf)]:
            key=(ds,seed,carrier,init_name)
            if key in done:continue
            ev=FeatureSubsetEvaluator(envs,mode='mean',n_neighbors=5,backend='sklearn');t0=time.perf_counter()
            if carrier=='NSGA-II':res=run_nsga2(ev.evaluate,D,a.pop,a.gen,random_state=seed,verbose=False,initial_population=init)
            else:res=run_spea2(ev.evaluate,D,a.pop,a.gen,archive_size=a.pop,random_state=seed,verbose=False,initial_population=init)
            search=time.perf_counter()-t0;train_hv=float(hypervolume_2d(res.pareto_objectives));h0=time.perf_counter();fm=front_test_metrics(res.pareto_population,envs,Xte,yte,5);held=time.perf_counter()-h0
            rec=dict(dataset=ds,seed=seed,carrier=carrier,init=init_name,n_samples=len(bundle.y),n_features=D,seed_sizes='/'.join(map(str,sizes)) if init_name=='MSF' else '',ranking_time=ranking_time if init_name=='MSF' else 0.0,train_hv=train_hv,**fm,unique_wrapper_evaluations=len(ev.cache),search_time=search,heldout_evaluation_time=held,total_time=(ranking_time if init_name=='MSF' else 0.0)+search+held,provenance='untouched_locked_dataset')
            rows.append(rec);done.add(key);pd.DataFrame(rows).to_csv(raw,index=False);write_stats(raw,a.out);print(json.dumps(rec),flush=True)
    write_stats(raw,a.out)

if __name__=='__main__':main()
