# P0-1 / P0-2 Frozen Confirmation Findings (Seeds 72–81)

## Status

- **Formal run complete: 120/120 records** = 4 datasets × 10 predeclared fresh seeds × 3 methods.
- **MSF-MOFS remained frozen throughout confirmation.** No algorithm, Fisher prior, guided-size schedule, mutation, archive, or Route-B/Cross-IR mechanism was changed in response to seeds 72–81.
- Smoke seed 72 is preserved separately for audit but is **excluded** from the formal 72–81 statistics because the HFMOEA timing executor was finalized after the smoke stage.
- IR={1,3,5} is treated only as the shared wrapper stress/evaluation protocol, not as part of MSF seeding.

## Frozen protocol audit

| Dataset | D | Frozen MSF guided sizes | Guided chromosomes | Population |
|---|---:|---:|---:|---:|
| MADELON | 500 | 10/25/50 | 3 | 12 |
| Hill-Valley | 100 | 5/10/20 | 3 | 12 |
| Musk1 | 166 | 8/10/17 | 3 | 12 |
| ARCENE | 10000 | 10/500/1000 | 3 | 12 |

## P0-1 — held-out Pareto-set HV confirmation

Values are mean ± sample standard deviation over paired seeds 72–81. Wilcoxon tests are two-sided; Holm correction is across the two predeclared MSF-vs-baseline comparisons within each dataset.

| Dataset | MSF-MOFS | HFMOEA-CP | MBLPE-CP | MSF vs HF: W/T/L; Holm p; r_rb | MSF vs MBLPE: W/T/L; Holm p; r_rb |
|---|---:|---:|---:|---|---|
| MADELON | 0.7575 ± 0.0144 | 0.7531 ± 0.0117 | 0.6058 ± 0.0444 | 3/5/2; 0.4375; 0.467 | 10/0/0; 0.003906; 1.000 |
| Hill-Valley | 0.3670 ± 0.0161 | 0.3160 ± 0.0219 | 0.3424 ± 0.0204 | 10/0/0; 0.003906; 1.000 | 8/0/2; 0.01953; 0.818 |
| Musk1 | 0.6907 ± 0.0348 | 0.6907 ± 0.0187 | 0.6845 ± 0.0367 | 5/0/5; 1; 0.018 | 6/0/4; 1; 0.236 |
| ARCENE | 0.6509 ± 0.0509 | 0.6295 ± 0.0873 | 0.5896 ± 0.0619 | 6/0/4; 0.5566; 0.236 | 9/0/1; 0.007812; 0.964 |

**Average rank across the four dataset-level mean PF-HVs:** MSF-MOFS = 1.00, HFMOEA-CP = 2.25, MBLPE-CP = 2.75.

### P0-1 interpretation

- The frozen method **passes the fresh-split confirmation gate**: MSF-MOFS has the best mean held-out PF-HV on all four datasets and retains average rank 1.00 on seeds 72–81.
- The result does **not** support a claim of universal pairwise statistical dominance. Against HFMOEA-CP, MADELON is not significant (Holm p=0.4375), Musk1 is effectively tied (mean difference ≈ 0.00003; Holm p=1.0), and ARCENE is not significant (Holm p≈0.5566). Hill-Valley is the clear significant MSF advantage.
- Against MBLPE-CP, MSF is significant on MADELON, Hill-Valley, and ARCENE under the within-dataset Holm correction, but not on Musk1.
- Regarding the reviewer concern about hidden MADELON tuning: because Full-data Fisher, the three guided sizes, and all other MSF choices were frozen before evaluating seeds 72–81, the new results materially weaken the hypothesis that the earlier MADELON result was merely a seed/split-specific artifact. However, these are still the same benchmark datasets, so this confirmation should **not** be described as proving complete dataset-level independence from prior benchmark exposure.

## P0-2 — wall-clock decomposition

Timing is method-specific elapsed wall-clock on the same environment. It excludes dataset loading, the outer split, and construction of the shared IR environments; it includes method preprocessing/filtering, wrapper search, and held-out Pareto-set evaluation.

### MSF-MOFS vs HFMOEA-CP

| Dataset | PF-HV: MSF / HF | Prep s: MSF / HF | Search s: MSF / HF | Total s: MSF / HF | Total reduction vs HF | Unique evals: MSF / HF | Total-time Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| MADELON | 0.7575 / 0.7531 | 0.010 / 2.283 | 5.319 / 7.250 | 5.362 / 9.562 | 43.9% | 75.9 / 78.6 | 0.007812 |
| Hill-Valley | 0.3670 / 0.3160 | 0.001 / 0.286 | 2.495 / 3.236 | 2.548 / 3.565 | 28.5% | 77.8 / 76.7 | 0.007812 |
| Musk1 | 0.6907 / 0.6907 | 0.001 / 0.309 | 2.320 / 2.440 | 2.420 / 2.827 | 14.4% | 77.2 / 81.7 | 0.1289 |
| ARCENE | 0.6509 / 0.6295 | 0.009 / 11.989 | 4.020 / 6.289 | 4.128 / 18.290 | 77.4% | 77.8 / 82.6 | 0.003906 |

### P0-2 interpretation

- **MADELON directly answers the reviewer’s efficiency question.** Held-out PF-HV is statistically indistinguishable between MSF and HFMOEA-CP (0.7575 vs 0.7531; Holm p=0.4375), while preprocessing falls from 2.283 s to 0.010 s and total method-specific time from 9.562 s to 5.362 s (44% lower mean total time).
- The preprocessing advantage is consistent: MSF has lower ranking/filter time on **10/10 seeds for every dataset**; paired Holm-adjusted p=0.00390625 for all four datasets. Absolute mean preprocessing savings versus HF are about 2.27 s (MADELON), 0.285 s (Hill-Valley), 0.308 s (Musk1), and 11.98 s (ARCENE).
- Total-time reduction versus HF is significant on MADELON, Hill-Valley, and ARCENE under the secondary timing tests; Musk1 shows a smaller 14% mean reduction and is not significant after Holm correction (p≈0.1289).
- **Wrapper-evaluation-budget efficiency and actual wall-clock efficiency are not the same claim.** MSF does not use fewer unique wrapper evaluations on every dataset (e.g., Hill-Valley mean 77.8 for MSF vs 76.7 for HF). The strongest new efficiency evidence is therefore the reduced preprocessing and measured elapsed time, not a universal reduction in evaluation count.
- MBLPE-CP also uses a single Fisher-style prefilter and therefore has very small preprocessing cost. The new result should not be phrased as “MSF is universally the fastest baseline”; the defensible claim is that MSF avoids the substantial ten-filter preprocessing overhead of HFMOEA-CP while retaining competitive or better held-out PF-HV under the common budget.

## Implementation / timing audit

- Exact old-protocol reproduction was checked before confirmation using MADELON seed 52: CISR, HFMOEA-CP, and MBLPE-CP reproduced the historical raw CSV values to numerical precision, validating the split, IR environments, 5-NN wrapper, NSGA-II adaptation, and PF-HV calculation.
- Under scikit-learn 1.8, repeated joblib/loky use for HFMOEA mutual-information filters caused process-pool instability in the container. The formal runner therefore uses an equivalent short-lived fork process pool for the per-feature MI calculations. On MADELON seed 72, the resulting MI score vector is **bit-for-bit identical** to `mutual_info_classif(..., n_jobs=1)` (max absolute difference 0.0; identical stable ranking).
- HFMOEA formal dataset/seed jobs were executed in fresh Python processes to prevent process-pool carry-over from contaminating timings. Observed slow seeds were retained; no trimming, best-of reruns, or timing cherry-picking was applied.
- `total_time` is internally consistent with preprocessing + search + held-out evaluation; maximum residual bookkeeping overhead across 120 rows is below 1 ms.

## Recommended manuscript/reviewer claim boundary

**Supported:** “With the MSF initializer frozen before evaluation, fresh seeds 72–81 preserve its rank-1 aggregate held-out PF-HV and show that a single Fisher ranking materially reduces preprocessing cost relative to HFMOEA-CP’s ten-filter guidance, including on MADELON where their PF-HV is statistically indistinguishable.”

**Do not claim:** universal statistical dominance; a universally safe Fisher prior; universally fewer wrapper evaluations; or absolute hardware-independent speedups.

## Remaining preregistered priorities

- P0-3 ARCENE mechanism analysis: Fisher rank stability + top-K overlap + population Hamming diversity (MADELON vs ARCENE; Random/Sparse-control/MSF).
- P0-4 large-population validation: population=50, exactly 3 guided chromosomes, comparable wrapper-evaluation budget; at least MADELON and preferably ARCENE.
