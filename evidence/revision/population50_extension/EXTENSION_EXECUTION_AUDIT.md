# Population-50 extension execution audit

## Completeness

- Hill-Valley: 10 Random + 10 MSF rows.
- Prostate-GE: 10 Random + 10 MSF rows.
- Seeds are exactly 72-81 for each dataset/method.
- No ARCENE partial row is included.

## Frozen execution environment

- Python: 3.13.5
- NumPy: 2.3.5
- pandas: 2.2.3
- SciPy: 1.17.0
- scikit-learn: 1.8.0

These versions match the archived `03_PUBLIC_MSF_SOURCE/ENVIRONMENT.txt`.

## Provenance

- Before-results protocol SHA-256: `85af70814272badffd423a798c08a8ad1137da901cb4d89ecd9edb9b0d62be3d`
- Extension runner SHA-256: `79a79e7d9a26f9578a6ce46c88d40d5ffb9d07be8b95d188846e652135f6d41c`
- Raw extension results SHA-256: `b8ac0a1ca6dcc9ea63381cc732c23b5bb111e8cd4714b27589d01f0aa787a439`
- Combined three-dataset statistics SHA-256: `4289079604d421270f733951ee13af77c6777f2b45ca9dc38acb74bbc6ac4c90`

The extension runner preserves the archived P0-4 evaluation/search logic and adds only generalized dataset loading/CLI paths for the two predeclared extension datasets. See `RUNNER_DIFF_FROM_ARCHIVED_P0_4.patch` for the exact source diff.

## Result integrity

The formal camera-ready comparison uses only complete paired 10-seed tables. Hill-Valley and Prostate-GE are complete; MADELON is imported from the archived formal P0-4 evidence. Holm correction in `large_population_three_dataset_combined.csv` is applied across the three population-50 datasets separately at Gen0 and Gen1.
