# MSF-MOFS v2 Final 9-Dataset Expansion Audit

Date: 2026-08-18

## Scope

This audit closes the pre-submission v2 expansion of the frozen `mixed_m3` MSF initializer. The expansion adds five datasets selected before any MSF result under a fixed ASU Biological Data rule, evaluates the same initializer under NSGA-II and SPEA2, and keeps the four-core-dataset external-baseline confirmation separate from the nine-dataset initializer breadth claim.

## Untouched breadth protocol

- Locked datasets: ALLAML, colon, GLI-85, Prostate-GE, SMK-CAN-187.
- Selection rule: ASU Biological Data; binary; D >= 2000; 50 <= N <= 200; exclude a duplicate leukemia representation; report all five without filtering.
- Seeds: 92-101.
- Population / generations: 12 / 6.
- Carriers: NSGA-II and SPEA2.
- Within each dataset/carrier: frozen MSF `mixed_m3` versus matched Random initialization.
- MSF: one full outer-training Fisher ranking; exactly 3 guided chromosomes; smallest three distinct integer sizes from {10, 0.05D, 0.10D, 0.20D}; standard downstream carrier thereafter.
- Outer split: stratified 70/30 with random_state = seed.
- Wrapper: fold-local scaling + 5-NN, stratified 3-fold, mean G-mean across IR={1,3,5}; feature-ratio second objective.
- Primary endpoint: held-out Pareto-set HV.
- Statistics: paired two-sided Wilcoxon; Holm across datasets separately within each carrier; rank-biserial effect size; wins/ties/losses. Carrier interaction compares paired MSF-minus-Random gains between NSGA-II and SPEA2.

The complete protocol and dataset SHA256 hashes are stored in `v2_untouched_breadth_final/PROTOCOL_BEFORE_RESULTS.json`.

## Completion

Formal untouched breadth run: **200/200 records complete** (5 datasets x 10 seeds x 2 carriers x 2 initializations).

All five locked datasets were retained and reported. No dataset was removed or substituted based on result quality.

## Five-new-dataset results

Across the five locked datasets, MSF beats matched Random on every seed under both carriers: **10 wins / 0 ties / 0 losses for all 10 dataset-carrier comparisons**. For each comparison, raw Wilcoxon p = 0.001953, rank-biserial = 1.0, and Holm-adjusted p across the five datasets within carrier = 0.009766.

Mean held-out PF-HV gains (MSF - Random):

| Dataset | NSGA-II | SPEA2 |
|---|---:|---:|
| ALLAML | +0.504275 | +0.521933 |
| colon | +0.387740 | +0.416425 |
| GLI-85 | +0.395452 | +0.380550 |
| Prostate-GE | +0.393618 | +0.396577 |
| SMK-CAN-187 | +0.161859 | +0.163855 |

No five-dataset carrier-interaction test is significant after Holm correction.

## Combined nine-dataset breadth result

Combining the four core datasets with the five untouched breadth datasets:

- MSF has higher dataset-level mean held-out PF-HV than matched Random on **9/9 datasets under NSGA-II** and **9/9 under SPEA2**.
- Seed-level paired outcomes: **87 wins / 0 ties / 3 losses per carrier**; **174 / 0 / 6** across both carriers.
- Dataset-level mean gain: **+0.274301 (NSGA-II)** and **+0.274333 (SPEA2)**.
- After Holm correction across the nine datasets separately within each carrier, **8/9 dataset comparisons are significant** under each carrier. Musk1 is positive but non-significant (Holm p = 0.083984).
- All nine carrier-interaction tests have Holm-adjusted p = 1.0; no carrier-dependent gain is detected in this protocol.
- Dataset-level average rank is MSF = 1.0 and Random = 2.0 under both carriers.

These findings support transfer across the **two tested binary MOEA carriers**; they do not establish arbitrary-MOEA generality.

## External-baseline evidence remains a separate claim

The strong-baseline comparison is intentionally restricted to the four core datasets. Final seeds 92-101 give dataset-level average rank MSF = 1.00, HFMOEA-CP = 2.25, MBLPE-CP = 2.75. The result is not described as universal pairwise dominance.

## Reproducibility audit

Earlier in the v2 expansion, MADELON seed=92 was independently re-run from the experiment runner for:

- NSGA-II + Random
- NSGA-II + MSF
- SPEA2 + Random
- SPEA2 + MSF

Train HV, held-out PF-HV, Pareto-set size, best held-out G-mean, minimum feature count, and unique wrapper evaluations matched the stored raw artifacts exactly. The external-baseline seed=92 re-run (MSF / HFMOEA-CP / MBLPE-CP) also matched all algorithmic result fields; only wall-clock naturally varied.

## Manuscript integration

The paper was revised to make two evidence roles explicit:

1. Four core datasets: sophisticated external-baseline comparison plus carrier validation.
2. Nine datasets: matched Random/MSF initializer breadth under NSGA-II and SPEA2.

The manuscript keeps the frozen `mixed_m3` rule; the new five datasets were not used for dose or schedule selection. Internet Advertisements and SECOM development-only references were removed from the final 10-page manuscript to avoid mixing development exposure with final breadth evidence.

Final manuscript version in this package: **v1.5**, 10 pages after render-and-inspect QA.

## Document QA

- Rendered pages: 10/10 inspected.
- Tables: 4.
- Native Word Math objects: 4.
- Figures: 1.
- Comments: 0.
- Tracked insertions/deletions: 0.
- Accessibility audit: 0 high / 0 medium / 0 low.
- No global font, margin, or table-font reduction was used to restore the 10-page limit.

## Remaining non-scientific submission blocker

The title page still contains author / affiliation / corresponding-email placeholders. These should be resolved only after confirming whether the actual EIC submission is anonymous.
