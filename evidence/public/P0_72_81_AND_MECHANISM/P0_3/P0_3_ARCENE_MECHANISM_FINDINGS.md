# P0-3 ARCENE Mechanism Analysis

## Scope

- Diagnostic only; **no MSF algorithm changes** were made.
- Outer seeds: 72–81.
- Fisher stability: all 45 pairwise outer-training comparisons per dataset.
- Population diversity: Random, Sparse-control, and frozen MSF-MOFS at generations 0, 1, 3, and 6; metric is mean pairwise normalized Hamming distance.
- Sparse-control uses the same matched base population and same three subset sizes as MSF, but permutes the Fisher-score-to-feature-identity mapping with the preregistered seed-based random permutation before forming top-k seeds.

## 1. Fisher ranking stability

| Dataset | Global Spearman rank correlation | Top-10 Jaccard | 5%D Jaccard | 10%D Jaccard | 20%D Jaccard |
|---|---:|---:|---:|---:|---:|
| MADELON | 0.428 ± 0.023 | 0.814 | 0.462 | 0.366 | 0.356 |
| ARCENE | 0.837 ± 0.037 | 0.181 | 0.604 | 0.709 | 0.773 |

**Key observation:** ARCENE does **not** show a globally random Fisher ordering. Its full-ranking Spearman correlation is actually higher than MADELON (≈0.837 vs ≈0.428). The instability is concentrated at the extreme top of the ranking: ARCENE Top-10 Jaccard is only ≈0.181 (median ≈0.053), whereas MADELON Top-10 Jaccard is ≈0.814. ARCENE becomes much more stable at broader sets (Top-500 ≈0.604; Top-1000 ≈0.709).

For the frozen MSF scales this matters directly: MADELON uses 10/25/50, while ARCENE uses 10/500/1000. Thus the ARCENE **ultra-sparse Top-10 seed is based on a much less stable feature identity set**, while the two broader ARCENE seeds are substantially more stable.

## 2. Population diversity

| Dataset | Method | Gen 0 | Gen 1 | Gen 3 | Gen 6 |
|---|---|---:|---:|---:|---:|
| MADELON | Random | 0.3217 | 0.2759 | 0.2124 | 0.1703 |
| MADELON | Sparse-control | 0.2759 | 0.1982 | 0.1113 | 0.0712 |
| MADELON | MSF-MOFS | 0.2759 | 0.1643 | 0.0351 | 0.0068 |
| ARCENE | Random | 0.3204 | 0.2794 | 0.2260 | 0.1638 |
| ARCENE | Sparse-control | 0.2717 | 0.1743 | 0.0962 | 0.0576 |
| ARCENE | MSF-MOFS | 0.2717 | 0.1696 | 0.0567 | 0.0190 |

At initialization, MSF and Sparse-control are essentially identical in Hamming diversity on both datasets (MADELON ≈0.27587 vs 0.27589; ARCENE ≈0.27166 vs 0.27169). This is expected because both replace exactly three matched individuals with nested subsets of identical cardinalities.

After selection begins, MSF contracts more strongly. By generation 6:
- MADELON: MSF 0.0068 vs Sparse-control 0.0712; MSF is lower on 10/10 seeds (descriptive paired Wilcoxon p=0.001953).
- ARCENE: MSF 0.0190 vs Sparse-control 0.0576; MSF is lower on 8/10 seeds (descriptive p=0.01367).

## Mechanistic conclusion

1. **The broad “D≫n makes the entire Fisher ranking unstable” hypothesis is not supported.** ARCENE has high full-ranking Spearman stability. A narrower statement is supported: the *very top* Fisher identities, especially Top-10, are unstable across outer splits.
2. **The “Fisher seeds reduce initialization diversity” hypothesis is not supported.** MSF and Sparse-control start with essentially the same Hamming diversity.
3. **Relevance guidance does induce faster downstream convergence**, but this is not an ARCENE-specific pathology: the diversity collapse is even more extreme on MADELON, where MSF is dramatically better than Sparse-control. Therefore low Hamming diversity by itself cannot explain the ARCENE negative result.
4. The most defensible interpretation is an **interaction** rather than a single cause: ARCENE’s ultra-sparse seed commits to a top-ranked feature set whose identities are unstable across outer splits, and selection can then concentrate the population around relevance-guided sparse regions. The present diagnostics support this interaction as plausible, but they do not establish a causal decomposition.

## Claim boundary for the paper

Supported: “ARCENE does not exhibit wholesale Fisher-rank instability, but its extreme Top-10 set is markedly less stable than MADELON’s. MSF and Sparse-control begin with matched diversity; MSF subsequently converges faster on both datasets, so diversity loss alone does not explain ARCENE. The negative result is most consistent with data-dependent uncertainty in the extreme Fisher prior interacting with rapid exploitation.”

Do not claim that Fisher is universally unsafe in D≫n data, or that population diversity alone caused the ARCENE result.
