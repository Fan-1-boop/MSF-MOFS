# Deterministic reconstruction of held-out Pareto-HV, mechanism family

**Source:** User-supplied original MSF source/evidence archive; SHA-256: `8be5bf83c786fb0d23c49190d5c59fc6244c8afde14d2ef3b66b3455969aeece`.

**Code:** Archived `03_PUBLIC_MSF_SOURCE/runners/run_p0_3_mechanism.py`, `irmofs/` modules and original core archives for MADELON and ARCENE. No modified optimizer, objective, subset definition, hyperparameter, or threshold. The original exploratory P0-3 implementation wrote `NaN` for most ARCENE held-out PF-HVs to avoid the added diagnostic runtime. It did not save final Pareto chromosomes. Accordingly, we **deterministically re-executed the original complete NSGA-II search**, then used the **original** `front_test_hv` evaluator on its final train-selected Pareto populations; this is a search replay, not recovery from a saved front.

**Protocol:** two binary public datasets (MADELON, ARCENE); seeds 72–81; matched Sparse-control and MSF-MOFS, Npop=12, 6 full generations, fixed feature sizes, outer 70/30 stratified test. Same training-only environments, fold-local StandardScaler + 5-NN, target IR set [1,3,5], held-out two-dimensional minimization HV ref=(1,1). Fisher stability: for each seed, mean Top-K Jaccard similarity with nine other outer-split Fisher rankings, averaged at the three actually injected scales: MADELON K=(10,25,50), ARCENE K=(10,500,1000). Each seed's held-out identity gain is HV(MSF) minus HV(Sparse-control).

**Exactness gates:** 40/40 method-run training HVs match the archive to within 5.6e-17; all 21 original non-null held-out HVs matched within 1e-9. Complete 10/10 matched seeds per dataset, no incomplete rows. Current Python 3.13.5; NumPy 2.3.5; SciPy 1.17.0; sklearn 1.8.0.

**Exploratory statistics (raw two-sided p-values, unadjusted):**

| Target | Dataset group | n | Spearman rho | p |
|:--|:--|--:|--:|--:|
| delta_train | pooled | 20 | +0.474 | 0.0349 |
| delta_train | madelon | 10 | -0.321 | 0.3655 |
| delta_train | arcene | 10 | -0.152 | 0.6761 |
| delta_heldout | pooled | 20 | +0.582 | 0.0071 |
| delta_heldout | madelon | 10 | -0.091 | 0.8028 |
| delta_heldout | arcene | 10 | +0.115 | 0.7514 |

**Key control:** dataset-centered pooled Spearman on held-out gain is ρ=0.021 (p=0.930). With only two independent datasets, the pooled n=20 observations are not 20 independent dataset replications. Accordingly the pooled association is descriptive and cannot establish within-dataset/per-run predictiveness.

**Held-out contrast:** MADELON mean +0.3536 (W/T/L 10/0/0, raw paired Wilcoxon p=0.001953; paired 95% bootstrap CI approximately [+0.334,+0.374]); ARCENE mean +0.0186 (6/0/4, p=0.375; bootstrap CI approximately [-0.012,+0.051]). CIs use 20,000 paired seed bootstrap resamples; fixed RNG seed 20261008; exploratory only.

**Files:** `reconstructed_heldout_runs.csv` (40 method-run raw), `reliability_heldout_paired_seed_level.csv` (20 same-seed pairs), `reliability_association_summary.csv`, `replay_p0_3_heldout.py` (adapt the ROOT path to your extracted source archive), and `stats_reliability_heldout.py`.
