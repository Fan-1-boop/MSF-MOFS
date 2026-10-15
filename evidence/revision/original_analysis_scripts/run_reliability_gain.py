from pathlib import Path
import pandas as pd, numpy as np
from scipy.stats import spearmanr
ROOT=Path('/mnt/data/eic_work/writing_pkg/MSF_SATELLITE_PAPER_WRITING_PACKAGE/02_ORIGINAL_MSF_EVIDENCE/03_RESULTS/P0_72_81_AND_MECHANISM/P0_3')
out=Path('/mnt/data/eic_work/revision_analysis')
pair=pd.read_csv(ROOT/'fisher_stability_pairwise.csv')
raw=pd.read_csv(ROOT/'population_diversity_raw.csv')
# actual three injected sizes per P0-3 dataset
sizes={'madelon':[10,25,50],'arcene':[10,500,1000]}
rows=[]
for ds,ks in sizes.items():
    q=pair[(pair.dataset==ds)&(pair.metric=='topk_jaccard')&pair.k.isin(ks)].copy()
    # each seed's reliability = mean Jaccard vs all other seeds, averaged over injected scales
    for seed in sorted(set(q.seed_a).union(set(q.seed_b))):
        vals=q[(q.seed_a==seed)|(q.seed_b==seed)].value.astype(float)
        rows.append({'dataset':ds,'seed':seed,'reliability_mean_jaccard':vals.mean(),'n_pair_scale_values':len(vals)})
rel=pd.DataFrame(rows)
# final hv one row/method/seed from generation 6
r=raw[raw.generation==6].pivot_table(index=['dataset','seed'],columns='method',values='test_pf_hv_final',aggfunc='first').reset_index()
r['identity_gain_hv']=r['MSF-MOFS']-r['Sparse-control']
df=rel.merge(r[['dataset','seed','identity_gain_hv']],on=['dataset','seed'])
# scale-specific reliabilities too
for ds,ks in sizes.items():
    for k in ks:
        q=pair[(pair.dataset==ds)&(pair.metric=='topk_jaccard')&(pair.k==k)]
        mp={}
        for seed in sorted(set(q.seed_a).union(set(q.seed_b))):
            mp[seed]=q[(q.seed_a==seed)|(q.seed_b==seed)].value.astype(float).mean()
        df.loc[df.dataset==ds,f'rel_k{k}']=df[df.dataset==ds].seed.map(mp)
df.to_csv(out/'prior_reliability_gain_seedlevel.csv',index=False)
res=[]
for ds,g in df.groupby('dataset'):
    rho,p=spearmanr(g.reliability_mean_jaccard,g.identity_gain_hv)
    res.append({'scope':ds,'n':len(g),'rho':rho,'p_two_sided':p,'mean_reliability':g.reliability_mean_jaccard.mean(),'mean_identity_gain_hv':g.identity_gain_hv.mean()})
# pooled raw
rho,p=spearmanr(df.reliability_mean_jaccard,df.identity_gain_hv)
res.append({'scope':'pooled_raw_2datasets','n':len(df),'rho':rho,'p_two_sided':p,'mean_reliability':df.reliability_mean_jaccard.mean(),'mean_identity_gain_hv':df.identity_gain_hv.mean()})
# pooled within-dataset ranks to remove gross dataset level shifts
z=df.copy(); z['rel_rank_within']=z.groupby('dataset').reliability_mean_jaccard.rank(pct=True);z['gain_rank_within']=z.groupby('dataset').identity_gain_hv.rank(pct=True)
rho,p=spearmanr(z.rel_rank_within,z.gain_rank_within)
res.append({'scope':'pooled_within_dataset_ranks','n':len(z),'rho':rho,'p_two_sided':p,'mean_reliability':np.nan,'mean_identity_gain_hv':np.nan})
res=pd.DataFrame(res);res.to_csv(out/'prior_reliability_gain_spearman.csv',index=False)
print(df.to_string(index=False));print('\n',res.to_string(index=False))
