# Five-Dataset Sparse-Control Attribution (NSGA-II, Seeds 92-101)

The control uses the same Bernoulli(0.20) base population and the same three frozen subset sizes as MSF. Fisher scores are permuted across feature identities with the pre-specified control RNG, preserving mixed-scale cardinality coverage while breaking relevance-informed feature identity. MSF is not retuned from these results.

| Dataset | Random | Sparse-control | MSF | Sparse-Random | MSF-Sparse |
|---|---:|---:|---:|---:|---:|
| ALLAML | 0.3441 | 0.4972 | 0.8484 | +0.1532 | +0.3511 |
| colon | 0.4540 | 0.5861 | 0.8417 | +0.1322 | +0.2556 |
| GLI-85 | 0.3993 | 0.5128 | 0.7948 | +0.1134 | +0.2820 |
| Prostate-GE | 0.4944 | 0.5639 | 0.8880 | +0.0696 | +0.3240 |
| SMK-CAN-187 | 0.4592 | 0.5472 | 0.6210 | +0.0880 | +0.0739 |

Sparse-control exceeds Random in mean held-out Pareto-set HV on all five datasets. MSF exceeds Sparse-control on all five. After Holm correction across the five datasets within each contrast, every Sparse-control-vs-Random comparison has p <= 0.027344 and every MSF-vs-Sparse-control comparison has p <= 0.013672. MSF-vs-Sparse wins/ties/losses range from 8/0/2 to 10/0/0.

This supports two empirically separable contributors under the primary NSGA-II carrier: mixed-scale sparse coverage and Fisher-guided feature identity. The older ARCENE control remains an important counterexample to any claim that Fisher identity is universally beneficial.
