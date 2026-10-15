# MSF-MOFS Final Core Confirmation (Seeds 92-101) — Corrected Audit Version

The final `mixed_m3` MSF configuration was frozen before seeds 92-101 were evaluated. The Hill-Valley label alias in the earlier packaged derivative files (`hill` vs `hill_valley`) has been normalized here; no algorithmic result value was changed.

## Held-out Pareto-set HV (mean ± std)

| Dataset | MSF-MOFS | HFMOEA-CP | MBLPE-CP |
|---|---:|---:|---:|
| MADELON | 0.755506 ± 0.009849 | 0.755449 ± 0.009836 | 0.629655 ± 0.044089 |
| Hill-Valley | 0.361617 ± 0.018843 | 0.320537 ± 0.021316 | 0.346387 ± 0.022909 |
| Musk1 | 0.707490 ± 0.023719 | 0.699566 ± 0.026934 | 0.690223 ± 0.028480 |
| ARCENE | 0.672849 ± 0.049902 | 0.605711 ± 0.060306 | 0.566476 ± 0.056669 |

Average dataset ranks: MSF-MOFS 1.00, HFMOEA-CP 2.25, MBLPE-CP 2.75.

## Paired inference

Holm correction is applied across the four core datasets separately within each baseline family.

### MSF-MOFS vs HFMOEA-CP
- MADELON: W/T/L = 1/9/0, raw p = 1.000000, Holm p = 1.000000.
- Hill-Valley: 10/0/0, raw p = 0.001953, Holm p = 0.007812.
- Musk1: 6/0/4, raw p = 0.275391, Holm p = 0.550781.
- ARCENE: 8/1/1, raw p = 0.007812, Holm p = 0.023438.

### MSF-MOFS vs MBLPE-CP
- MADELON: W/T/L = 10/0/0, raw p = 0.001953, Holm p = 0.007812.
- Hill-Valley: 9/0/1, raw p = 0.027344, Holm p = 0.054688.
- Musk1: 7/0/3, raw p = 0.193359, Holm p = 0.193359.
- ARCENE: 8/0/2, raw p = 0.009766, Holm p = 0.029297.

## Interpretation boundary

Numerical mean leadership and average rank are reported separately from pairwise statistical significance. The final core confirmation does not support universal pairwise dominance. Timing is hardware dependent and is retained alongside unique wrapper-evaluation counts.
