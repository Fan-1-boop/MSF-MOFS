# Population-50 extension: three-dataset formal camera-ready result

Protocol: NSGA-II, seeds 72-81, population 50, exactly three MSF-guided chromosomes (6% occupancy), unchanged frozen MSF definition, and held-out Pareto-front HV. Generation 0 and one complete generation are reported to bracket the original population-12 evaluation budget without introducing a partial NSGA-II generation.

| Stage | Dataset | Guided sizes | Random HV | MSF HV | mean ΔHV [95% CI] | W/T/L | RBC | Holm p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gen0 | MADELON | 10/25/50 | 0.3260 ± 0.0117 | 0.7535 ± 0.0105 | +0.4274 [+0.4181, +0.4368] | 10/0/0 | 1.000 | 0.005859 |
| gen0 | Hill-Valley | 5/10/20 | 0.3207 ± 0.0116 | 0.3422 ± 0.0149 | +0.0215 [+0.0175, +0.0252] | 10/0/0 | 1.000 | 0.005859 |
| gen0 | Prostate-GE | 10/298/597 | 0.4513 ± 0.0763 | 0.8849 ± 0.0617 | +0.4336 [+0.3890, +0.4828] | 10/0/0 | 1.000 | 0.005859 |
| gen1 | MADELON | 10/25/50 | 0.3370 ± 0.0146 | 0.7535 ± 0.0105 | +0.4165 [+0.4064, +0.4271] | 10/0/0 | 1.000 | 0.005859 |
| gen1 | Hill-Valley | 5/10/20 | 0.3226 ± 0.0142 | 0.3538 ± 0.0159 | +0.0312 [+0.0224, +0.0424] | 10/0/0 | 1.000 | 0.005859 |
| gen1 | Prostate-GE | 10/298/597 | 0.4632 ± 0.0726 | 0.8849 ± 0.0617 | +0.4217 [+0.3840, +0.4591] | 10/0/0 | 1.000 | 0.005859 |

## Interpretation

- All three datasets show 10/10 paired MSF wins at both generation 0 and generation 1.
- At generation 1, mean held-out HV gains are +0.4165 on MADELON, +0.0312 on Hill-Valley, and +0.4217 on Prostate-GE.
- The result spans D=100, 500, and 5,966 and sample sizes N=1,212, 2,600, and 102, respectively, while the guided count remains fixed at three (6% of population 50).
- The evidence therefore addresses the population-occupancy concern without changing the algorithm: the effect persists when the guided fraction falls from 25% (3/12) to 6% (3/50), including two additional datasets with contrasting dimensional/sample regimes.
- ARCENE remains excluded because the prior attempt did not yield a complete paired 10-seed table within its execution window; no incomplete ARCENE rows are used.
