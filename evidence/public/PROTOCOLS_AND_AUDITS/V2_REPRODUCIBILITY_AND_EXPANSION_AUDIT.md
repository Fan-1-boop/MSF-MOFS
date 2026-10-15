# MSF-MOFS v2 Reproducibility and Expansion Audit

Date: 2026-08-18

## Scope

This audit covers the pre-submission v2 expansion of MSF-MOFS. The expansion separates development from final confirmation and does not treat previously exposed datasets as untouched validation.

## 1. Development / freeze / confirmation provenance

- Development seeds: 82–91.
- Development tasks: cross-MOEA carrier validation and guidance-dose / sparsity-schedule sensitivity.
- Candidate configurations: Random, mixed_m1, mixed_m2, mixed_m3, mixed_m4, relative_m3.
- Selected final configuration: mixed_m3.
- Change relative to v1.1 final MSF algorithm: none.
- Final confirmation seeds: 92–101.
- Freeze record: `v2_development_summary/FINAL_V2_SELECTION_FREEZE.json`.
- No changes to the final MSF configuration were made after inspection of seeds 92–101.

The retained final MSF configuration remains: one full outer-training Fisher ranking; three mixed absolute-relative sparse seeds; exactly three replacements in a matched random initial population; standard downstream binary MOEA thereafter.

## 2. Final external-baseline confirmation (seeds 92–101)

Held-out Pareto-set HV, mean ± std:

| Dataset | MSF-MOFS | HFMOEA-CP | MBLPE-CP |
|---|---:|---:|---:|
| MADELON | 0.755506 ± 0.009849 | 0.755449 ± 0.009836 | 0.629655 ± 0.044089 |
| Hill-Valley | 0.361617 ± 0.018843 | 0.320537 ± 0.021316 | 0.346387 ± 0.022909 |
| Musk1 | 0.707490 ± 0.023719 | 0.699566 ± 0.026934 | 0.690223 ± 0.028480 |
| ARCENE | 0.672849 ± 0.049902 | 0.605711 ± 0.060306 | 0.566476 ± 0.056669 |

Average ranks: MSF-MOFS 1.00, HFMOEA-CP 2.25, MBLPE-CP 2.75.

The result does not justify universal pairwise statistical dominance. In particular, MSF vs HFMOEA-CP is effectively tied on MADELON and is not significant on Musk1.

## 3. Cross-MOEA carrier validation

The same frozen MSF initializer was evaluated with matched Random initialization under NSGA-II and SPEA2.

Core-dataset mean held-out PF-HV gains (MSF − Random):

| Dataset | NSGA-II | SPEA2 |
|---|---:|---:|
| MADELON | +0.407676 | +0.405776 |
| Hill-Valley | +0.025704 | +0.034939 |
| Musk1 | +0.029285 | +0.026710 |
| ARCENE | +0.163097 | +0.122234 |

No carrier-by-initialization interaction was detected after Holm correction on the four core datasets. This supports transfer across the two tested carriers, not arbitrary-MOEA universality.

Internet Advertisements and SECOM were also replicated on new seeds under both carriers, but they are explicitly retained as pre-exposed replication datasets rather than untouched validation datasets.

## 4. Guidance-dose / schedule development study

Development seeds 82–91 compared Random, mixed m=1/2/3/4, and relative-only m=3. Across six development datasets, mixed_m3 was not consistently statistically inferior to any alternative and was retained without modification. Relative_m3 had the best numerical average rank but was significantly worse than mixed_m3 on Internet Advertisements, so no post-hoc switch was made.

This study supports robustness of the three-seed mixed schedule. It does not establish theoretical optimality of three seeds or of the exact scale values.

## 5. Classifier sensitivity

A post-freeze diagnostic used fold-local StandardScaler + LinearSVC (`max_iter=20000`) on MADELON, Hill-Valley, and ARCENE. MSF − Random mean PF-HV gains were positive on all three datasets, with significant gains on MADELON and ARCENE after Holm correction. The effect is therefore not unique to the 5-NN wrapper, while remaining learner-dependent.

## 6. Independent reproduction checks

Two independent seed-92 reruns were performed from the runners and raw datasets.

### Cross-MOEA runner

MADELON seed 92 was rerun for:
- NSGA-II + Random
- NSGA-II + MSF
- SPEA2 + Random
- SPEA2 + MSF

The rerun matched the archived raw results exactly for train HV, held-out PF-HV, Pareto-set size, best held-out mean G-mean, minimum selected-feature count, and unique wrapper evaluations. Wall-clock values were not expected to match exactly.

### External-baseline runner

MADELON seed 92 was rerun for MSF-MOFS, HFMOEA-CP, and MBLPE-CP. The rerun matched archived raw results exactly for all algorithmic/result fields checked, including train HV, held-out PF-HV, Pareto-set size, best/worst held-out G-mean, minimum feature count, and unique wrapper evaluations. Timing varied as expected.

## 7. Raw-artifact inventory and data hygiene

Key raw files:

- `v2_final_confirm_core/confirmation_raw.csv`: 120 rows (4 datasets × 10 seeds × 3 methods).
- `v2_final_confirm_core/confirmation_raw_normalized.csv`: normalized dataset aliases for downstream statistics.
- `v2_final_plugin_core/raw.csv`: 160 rows (4 datasets × 10 seeds × 2 carriers × 2 initializers).
- `v2_dose_core/raw.csv`: 240 rows.
- `v2_dose_natural/raw.csv`: 120 rows.
- `v2_classifier_sensitivity_final/raw.csv`: 60 rows.
- `v2_final_plugin_6dataset_raw.csv`: 240 rows.

The original confirmation raw file contains mixed aliases (`hill` / `hill_valley`, `musk`); the normalized raw file resolves these aliases before aggregation. This is a metadata hygiene issue only and does not alter the numerical results.

## 8. Locked untouched biological-dataset expansion

Before outcome inspection, five additional biological datasets were locked using an objective ASU/scikit-feature rule: binary labels, D >= 2000, 50 <= N <= 200, with duplicate leukemia cohort exclusion.

Locked set:
- ALLAML (72 × 7129)
- colon (62 × 2000)
- GLI-85 (85 × 22283)
- Prostate-GE (102 × 5966)
- SMK-CAN-187 (187 × 19993)

These five datasets have not yet been executed because the current runtime cannot materialize the `.mat` binary files from the resolved web/GitHub sources. No replacement dataset has been substituted. They remain the only admissible untouched expansion set under the locked protocol.

## 9. Manuscript integration

The v2 evidence is integrated into manuscript v1.3:
- final confirmation uses seeds 92–101;
- NSGA-II / SPEA2 carrier validation is reported;
- dose/schedule development evidence is bounded to development seeds;
- Internet Ads and SECOM are labeled pre-exposed new-seed replications;
- LinearSVC is presented as a diagnostic, not a new formal benchmark axis;
- Fig. 1 now identifies a downstream binary MOEA (NSGA-II / SPEA2) rather than implying NSGA-II-only binding;
- table standard deviations use consistent leading-zero formatting.

The rendered manuscript remains 10 pages. All ten pages were visually inspected. Accessibility audit: 0 high, 0 medium, 0 low findings. No comments are present; no tracked-change insertion/deletion elements are present.

## 10. Current decision

The v2 evidence strengthens the paper without increasing algorithmic complexity. The retained algorithm remains mixed_m3. The main remaining experimental block is the predeclared five-dataset untouched ASU expansion; it should be run only on those locked files, not on substitutes selected after seeing results.
