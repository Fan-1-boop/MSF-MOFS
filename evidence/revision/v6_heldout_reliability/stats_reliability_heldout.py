import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr,wilcoxon
ROOT=Path('/mnt/data/heldout_audit_work/MSF_MOFS_EIC2026_PAPER_AND_SATELLITE_VALIDATION_PACKAGE')
pair=pd.read_csv(ROOT/'05_PUBLIC_MSF_EVIDENCE/P0_72_81_AND_MECHANISM/P0_3/fisher_stability_pairwise.csv')
run=pd.read_csv('/mnt/data/heldout_audit_work/reconstructed_heldout_runs.csv')
assert len(run)==40 and not run.duplicated(['dataset','seed','method']).any() and run.test_pf_hv_replayed.notna().all()
assert run.groupby(['dataset','method']).size().eq(10).all()
assert run.train_hv_abs_error.max()<1e-12
r=[]
for ds in ['madelon','arcene']:
 dd=pair[(pair.dataset==ds)&(pair.metric=='topk_jaccard')]; sizes={'madelon':{10,25,50},'arcene':{10,500,1000}}[ds]
 assert sizes.issubset(set(dd.k.dropna().astype(int).unique()))
 dd=dd[dd.k.isin(sizes)]
 for seed in range(72,82):
  ss=dd[(dd.seed_a==seed)|(dd.seed_b==seed)]
  assert len(ss)==27,(ds,seed,len(ss))
  # equal weight over each of the three K scales
  reli=ss.groupby('k').value.mean().mean()
  sub=run[(run.dataset==ds)&(run.seed==seed)].set_index('method')
  assert set(sub.index)=={'Sparse-control','MSF-MOFS'}
  r.append(dict(dataset=ds,seed=seed,reliability_topk_jaccard=reli,
    hv_train_msf=sub.loc['MSF-MOFS','train_hv_replayed'],hv_train_sparse=sub.loc['Sparse-control','train_hv_replayed'],
    hv_test_msf=sub.loc['MSF-MOFS','test_pf_hv_replayed'],hv_test_sparse=sub.loc['Sparse-control','test_pf_hv_replayed'],
    delta_train=sub.loc['MSF-MOFS','train_hv_replayed']-sub.loc['Sparse-control','train_hv_replayed'],
    delta_heldout=sub.loc['MSF-MOFS','test_pf_hv_replayed']-sub.loc['Sparse-control','test_pf_hv_replayed']))
out=pd.DataFrame(r)
out.to_csv('/mnt/data/heldout_audit_work/reliability_heldout_paired_seed_level.csv',index=False)
print('REPLAY n=',len(run),'max absolute trainHV error=',run.train_hv_abs_error.max(),'n of prior nonmissing test matches',run.test_pf_hv_archived.notna().sum())
print('DIAGNOSTIC columns:',list(out.columns));print(out.round(4).to_string(index=False))
for v in ['delta_train','delta_heldout']:
 print('\nTARGET',v)
 for ds, dd in [('pooled',out),*list(out.groupby('dataset'))]:
  rho,p=spearmanr(dd.reliability_topk_jaccard,dd[v])
  dx=dd[v].to_numpy()
  print(ds,'n',len(dd),'rho',rho,'p',p,'mean reliability',dd.reliability_topk_jaccard.mean(),'mean delta',dd[v].mean(),'median delta',np.median(dx),'w/t/l',sum(dx>0),sum(dx==0),sum(dx<0))
  if ds!='pooled':
   print('paired wilcoxon two-sided raw p',wilcoxon(dx).pvalue)
# isolate between dataset effect
for v in ['delta_train','delta_heldout']:
 x=out.reliability_topk_jaccard-out.groupby('dataset').reliability_topk_jaccard.transform('mean')
 y=out[v]-out.groupby('dataset')[v].transform('mean')
 print('dataset-centered pooled spearman',v,spearmanr(x,y))
# generate report summary csv
rows=[]
for target in ['delta_train','delta_heldout']:
 for name,df in [('pooled',out),('madelon',out.query('dataset=="madelon"')),('arcene',out.query('dataset=="arcene"'))]:
  rho,p=spearmanr(df.reliability_topk_jaccard,df[target]);rows.append(dict(outcome=target,group=name,n=len(df),rho=rho,p_two_sided=p,mean_delta=df[target].mean(),min_delta=df[target].min(),max_delta=df[target].max(),wins=(df[target]>0).sum(),ties=(df[target]==0).sum(),losses=(df[target]<0).sum()))
pd.DataFrame(rows).to_csv('/mnt/data/heldout_audit_work/reliability_association_summary.csv',index=False)
