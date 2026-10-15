# MSF-MOFS v2: Final 9-Dataset Cross-MOEA Breadth Validation

## Protocol

- Final MSF configuration: frozen `mixed_m3` (one full outer-training Fisher ranking; exactly three guided chromosomes; smallest three distinct sizes from `{10, 0.05D, 0.10D, 0.20D}`); no method change is allowed from this breadth study.
- Seeds: 92–101.
- Carriers: NSGA-II and SPEA2.
- Control: matched Random initialization using the same base population and downstream RNG seed.
- Wrapper: 5-NN, fold-local scaling, 3-fold validation, mean G-mean over IR={1,3,5}; feature-ratio second objective.
- Primary endpoint: held-out Pareto-set HV after a stratified 70/30 outer split.
- Five new biological datasets were locked before any MSF result using the rule: ASU Biological Data, binary, D>=2000, 50<=N<=200, excluding a duplicate leukemia representation. All five are reported.
- Inference: paired two-sided Wilcoxon MSF vs Random, Holm correction across the nine datasets separately within each carrier; rank-biserial effect size and wins/ties/losses. Carrier-interaction tests compare paired MSF-minus-Random gains and use Holm across nine datasets.

## Held-out PF-HV means

| Dataset | NSGA-II Random | NSGA-II MSF | Gain | SPEA2 Random | SPEA2 MSF | Gain |
|---|---:|---:|---:|---:|---:|---:|
| madelon | 0.3478 | **0.7555** | +0.4077 | 0.3529 | **0.7587** | +0.4058 |
| hill_valley | 0.3359 | **0.3616** | +0.0257 | 0.3331 | **0.3680** | +0.0349 |
| musk1 | 0.6782 | **0.7075** | +0.0293 | 0.6822 | **0.7090** | +0.0267 |
| arcene | 0.5098 | **0.6728** | +0.1631 | 0.5206 | **0.6428** | +0.1222 |
| ALLAML | 0.3441 | **0.8484** | +0.5043 | 0.3332 | **0.8552** | +0.5219 |
| colon | 0.4540 | **0.8417** | +0.3877 | 0.4326 | **0.8490** | +0.4164 |
| GLI-85 | 0.3993 | **0.7948** | +0.3955 | 0.4167 | **0.7972** | +0.3806 |
| Prostate-GE | 0.4944 | **0.8880** | +0.3936 | 0.4925 | **0.8891** | +0.3966 |
| SMK-CAN-187 | 0.4592 | **0.6210** | +0.1619 | 0.4584 | **0.6222** | +0.1639 |

## Paired inference

### NSGA-II

| Dataset | Mean gain | W/T/L | r_rb | p raw | p Holm(9) |
|---|---:|---:|---:|---:|---:|
| madelon | +0.4077 | 10/0/0 | 1.000 | 0.001953 | 0.017578 |
| hill_valley | +0.0257 | 9/0/1 | 0.855 | 0.013672 | 0.027344 |
| musk1 | +0.0293 | 8/0/2 | 0.636 | 0.083984 | 0.083984 |
| arcene | +0.1631 | 10/0/0 | 1.000 | 0.001953 | 0.017578 |
| ALLAML | +0.5043 | 10/0/0 | 1.000 | 0.001953 | 0.017578 |
| colon | +0.3877 | 10/0/0 | 1.000 | 0.001953 | 0.017578 |
| GLI-85 | +0.3955 | 10/0/0 | 1.000 | 0.001953 | 0.017578 |
| Prostate-GE | +0.3936 | 10/0/0 | 1.000 | 0.001953 | 0.017578 |
| SMK-CAN-187 | +0.1619 | 10/0/0 | 1.000 | 0.001953 | 0.017578 |

### SPEA2

| Dataset | Mean gain | W/T/L | r_rb | p raw | p Holm(9) |
|---|---:|---:|---:|---:|---:|
| madelon | +0.4058 | 10/0/0 | 1.000 | 0.001953 | 0.017578 |
| hill_valley | +0.0349 | 10/0/0 | 1.000 | 0.001953 | 0.017578 |
| musk1 | +0.0267 | 7/0/3 | 0.636 | 0.083984 | 0.083984 |
| arcene | +0.1222 | 10/0/0 | 1.000 | 0.001953 | 0.017578 |
| ALLAML | +0.5219 | 10/0/0 | 1.000 | 0.001953 | 0.017578 |
| colon | +0.4164 | 10/0/0 | 1.000 | 0.001953 | 0.017578 |
| GLI-85 | +0.3806 | 10/0/0 | 1.000 | 0.001953 | 0.017578 |
| Prostate-GE | +0.3966 | 10/0/0 | 1.000 | 0.001953 | 0.017578 |
| SMK-CAN-187 | +0.1639 | 10/0/0 | 1.000 | 0.001953 | 0.017578 |

## Main findings

- Dataset-level mean PF-HV favors MSF on 9/9 datasets under NSGA-II and 9/9 under SPEA2. Average dataset rank is 1.00 for MSF and 2.00 for Random under both carriers.
- Across the 90 matched seed comparisons per carrier, NSGA-II gives 87/0/3 wins/ties/losses and SPEA2 gives 87/0/3.
- Mean dataset-level PF-HV gain is +0.2743 under NSGA-II and +0.2743 under SPEA2.
- After Holm correction across all nine datasets within each carrier, 8/9 datasets are significant for each carrier; Musk1 remains positive but non-significant.
- All nine carrier-interaction tests are non-significant after Holm correction (all adjusted p=1.0), so no dataset provides evidence that the MSF gain depends on NSGA-II versus SPEA2 within this protocol.
- On the five newly materialized untouched biological datasets alone, MSF records 10/10 wins on every dataset under both carriers, with rank-biserial effect size 1.0 and Holm-adjusted p=0.009766 within the five-dataset family.

## Claim boundary

- This establishes empirical transfer across two tested binary MOEA carriers and a nine-dataset high-dimensional suite; it does not prove arbitrary-MOEA generality.
- Sophisticated external baselines HFMOEA-CP and MBLPE-CP remain evaluated on the four core datasets. The five new datasets are a breadth/control study of the initializer itself, not an expanded state-of-the-art baseline tournament.
- The five added datasets are all small-n/high-D biological datasets, so breadth increases strongly in dimensional regime but not uniformly in application domain.
- No result from these five datasets was used to modify the frozen MSF configuration.