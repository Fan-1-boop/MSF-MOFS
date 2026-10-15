# MSF-MOFS Reviewer-Driven P0 Evidence — Frozen Follow-up

Date: 2026-08-17

## Research-discipline status

The final MSF-MOFS algorithm remained frozen throughout this follow-up. No Cross-IR mechanism, stability penalty, mutation change, archive change, adaptive prior selection, or seed-size retuning was introduced after observing seeds 72–81. Route B was not restarted.

## P0-1 — Fresh frozen confirmation (seeds 72–81)

Formal design: 4 datasets × 10 paired fresh seeds × 3 methods = **120/120 complete records**.

Held-out Pareto-set HV (mean ± std):

| Dataset | MSF-MOFS | HFMOEA-CP | MBLPE-CP |
|---|---:|---:|---:|
| MADELON | **0.7575 ± 0.0144** | 0.7531 ± 0.0117 | 0.6058 ± 0.0444 |
| Hill-Valley | **0.3670 ± 0.0161** | 0.3160 ± 0.0219 | 0.3424 ± 0.0204 |
| Musk1 | **0.6907 ± 0.0348** | 0.6907 ± 0.0187 | 0.6845 ± 0.0367 |
| ARCENE | **0.6509 ± 0.0509** | 0.6295 ± 0.0873 | 0.5896 ± 0.0619 |

Average ranks: **MSF-MOFS 1.00**, HFMOEA-CP 2.25, MBLPE-CP 2.75.

Primary paired results preserve the required claim boundary: MSF is not universally statistically superior. Versus HFMOEA-CP, MADELON (Holm p=0.4375), Musk1 (p=1.0), and ARCENE (p≈0.5566) are not significant; Hill-Valley is significant. The fresh frozen splits therefore support generalization of the selected Full-data Fisher design across unseen splits, but they do not prove complete benchmark-level independence from earlier exposure to the same datasets.

## P0-2 — Wall-clock decomposition

Method-specific timing excludes loading/split/shared IR-environment construction and includes filter/ranking preprocessing, wrapper search, and held-out Pareto-set evaluation.

MSF vs HFMOEA-CP mean times:

| Dataset | Prep: MSF / HF (s) | Total: MSF / HF (s) | Total reduction |
|---|---:|---:|---:|
| MADELON | 0.010 / 2.283 | 5.362 / 9.562 | 43.9% |
| Hill-Valley | 0.001 / 0.286 | 2.548 / 3.565 | 28.5% |
| Musk1 | 0.001 / 0.309 | 2.420 / 2.827 | 14.4% |
| ARCENE | 0.009 / 11.989 | 4.128 / 18.290 | 77.4% |

On every dataset, MSF preprocessing is lower on 10/10 seeds (Holm-adjusted paired p=0.00390625). Total time is significantly lower on MADELON, Hill-Valley, and ARCENE in the secondary timing analysis, but not Musk1 after Holm correction.

The important distinction is preserved: **wrapper-evaluation-budget efficiency is not identical to wall-clock efficiency**. MSF does not always use fewer unique wrapper evaluations than HFMOEA-CP (Hill-Valley is a counterexample), but it avoids the ten-filter preprocessing overhead.

For the reviewer's MADELON question specifically, PF-HV is statistically indistinguishable (0.7575 vs 0.7531) while MSF reduces mean preprocessing from 2.283 s to 0.010 s and total method-specific time from 9.562 s to 5.362 s.

## P0-3 — ARCENE mechanism analysis

### Fisher stability

Across all 45 pairwise comparisons of outer-training rankings from seeds 72–81:

| Dataset | Global Spearman | Top-10 Jaccard | 5%D Jaccard | 10%D Jaccard |
|---|---:|---:|---:|---:|
| MADELON | 0.428 | **0.814** | 0.462 | 0.366 |
| ARCENE | **0.837** | **0.181** | 0.604 | 0.709 |

This rejects the simple story that the entire ARCENE Fisher ranking is unstable. The sharper phenomenon is **extreme-top instability**: ARCENE's Top-10 identity set changes substantially across outer splits (median Jaccard ≈0.053), whereas its broader Top-500/Top-1000 sets are much more stable. This directly affects the frozen ARCENE seed schedule 10/500/1000.

### Population diversity

Mean pairwise normalized Hamming distance:

| Dataset | Method | Gen 0 | Gen 1 | Gen 3 | Gen 6 |
|---|---|---:|---:|---:|---:|
| MADELON | Random | 0.3217 | 0.2759 | 0.2124 | 0.1703 |
| MADELON | Sparse-control | 0.2759 | 0.1982 | 0.1113 | 0.0712 |
| MADELON | MSF | 0.2759 | 0.1643 | 0.0351 | 0.0068 |
| ARCENE | Random | 0.3204 | 0.2794 | 0.2260 | 0.1638 |
| ARCENE | Sparse-control | 0.2717 | 0.1743 | 0.0962 | 0.0576 |
| ARCENE | MSF | 0.2717 | 0.1696 | 0.0567 | 0.0190 |

MSF and Sparse-control start with essentially identical diversity, so Fisher-guided feature identities do **not** reduce initialization diversity by construction. MSF then converges faster. However, the downstream diversity collapse is even stronger on MADELON, where relevance guidance is highly beneficial. Therefore low Hamming diversity alone cannot explain ARCENE's historical Sparse-control > MSF result.

The defensible mechanism statement is an interaction: ARCENE's ultra-sparse Top-10 Fisher identities are unstable, while relevance-guided search can rapidly concentrate around the guided sparse region. These diagnostics make that interaction plausible without establishing a single causal explanation.

## P0-4 — Large-population validation

Completed formal minimum requirement: **MADELON, seeds 72–81, population=50, exactly 3 guided chromosomes (6%)**.

To preserve standard NSGA-II, no partial generation was introduced. The original ~70–83 unique-evaluation budget lies between the two feasible standard-generation brackets for N=50: initialization (~50) and one complete generation (~98).

| Stage | Random PF-HV | MSF PF-HV | Mean evals Random / MSF | W/T/L | p |
|---|---:|---:|---:|---:|---:|
| Gen 0 | 0.3260 ± 0.0117 | **0.7535 ± 0.0105** | 50.0 / 50.0 | 10/0/0 | 0.001953 |
| Gen 1 | 0.3370 ± 0.0146 | **0.7535 ± 0.0105** | 98.1 / 98.2 | 10/0/0 | 0.001953 |

Thus the MADELON advantage survives when the guided fraction falls from 25% to 6%. This directly rejects the explanation that the original effect is merely due to 3 guided chromosomes occupying too much of a population of 12.

ARCENE was attempted as the optional second large-population dataset, but pop=50 held-out Pareto-set evaluation at D=10,000 exceeded the single-job execution window before a complete 10-seed paired table was obtained. No partial ARCENE result is used.

## Overall reviewer-facing conclusion

The strongest revised evidence package is now:

1. a **frozen fresh-seed confirmation** that preserves MSF's rank-1 aggregate result without universal-dominance claims;
2. direct **wall-clock evidence** that a single Fisher ranking is substantially cheaper than HFMOEA-CP's ten-filter preprocessing, especially on MADELON and ARCENE;
3. a more precise ARCENE limitation: not wholesale Fisher instability, but instability concentrated at the extreme top of the ranking plus rapid relevance-guided convergence;
4. a **population=50 control** showing that the MADELON effect persists when guided chromosomes are only 6% of the population.

The algorithm should remain frozen. These results strengthen the paper by improving evidence and claim calibration rather than adding mechanisms.
