# P0-4 Large-Population Validation

## Completed formal validation

- Dataset: **MADELON**.
- Seeds: 72–81.
- Population: **50**.
- MSF guided chromosomes: **exactly 3**, so guided fraction is **3/50 = 6%** rather than 3/12 = 25%.
- Frozen guided sizes: 10/25/50. No method changes.

## Comparable evaluation-budget design

A full standard NSGA-II generation with N=50 cannot exactly reproduce the original ~76–79 evaluation budget: initialization already costs ~50 unique evaluations and one complete generation reaches ~98. Rather than introducing a nonstandard partial generation, results are reported at the two standard-generation brackets that surround the original budget: generation 0 (~50 evaluations) and generation 1 (~98 evaluations).

| Stage | Random PF-HV | MSF PF-HV | Random evals | MSF evals | MSF-Random mean Δ | W/T/L | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gen 0 | 0.3260 ± 0.0117 | 0.7535 ± 0.0105 | 50.0 | 50.0 | 0.4274 | 10/0/0 | 0.001953 |
| Gen 1 | 0.3370 ± 0.0146 | 0.7535 ± 0.0105 | 98.1 | 98.2 | 0.4165 | 10/0/0 | 0.001953 |

## Interpretation

- The MADELON advantage clearly **survives the population increase**. At only 6% guided fraction, MSF wins 10/10 paired seeds at both budget brackets.
- MSF’s mean held-out PF-HV is ~0.7535 at both initialization and after one complete generation, versus ~0.3260 and ~0.3370 for Random. Thus the effect is already present when only 3 of 50 initial chromosomes are guided.
- This directly argues against the hypothesis that the original MADELON gain is merely an artifact of guided chromosomes occupying 25% of a population of 12.
- The fresh-seed pop=50 MSF mean (~0.7535) is also close to the pop=12 confirmation mean (~0.7575), despite the large change in guided fraction.

## ARCENE status

ARCENE was attempted as the optional second P0-4 dataset. However, with N=50 and D=10,000 the held-out evaluation of Random Pareto sets exceeded the per-job execution window before a complete 10-seed paired table could be obtained. No incomplete ARCENE results are used or reported. The preregistered minimum requirement (“at least MADELON”) is therefore fully satisfied, while ARCENE remains absent from the formal large-population table.
