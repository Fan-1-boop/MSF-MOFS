# Satellite MSF Feature-Selection Validation Summary

This directory contains only the satellite **feature-selection/search** validation used to support the conference paper. Satellite health-estimation algorithms (evidence fusion, smoothing, DBN/health-state inference, thresholds, and health-service code) are intentionally excluded.

## Primary comparison used in the paper

Matched MSF vs Random, search seeds 192--201, same wrapper/objectives/search budget.

| Subsystem | D | MSF HV | Random HV | Delta HV | W/T/L | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| Power | 267 | 0.8054 +/- 0.0069 | 0.7222 +/- 0.0120 | +0.0832 | 10/0/0 | 0.007812 |
| Thermal | 135 | 0.8807 +/- 0.0161 | 0.8343 +/- 0.0200 | +0.0464 | 10/0/0 | 0.007812 |
| Attitude | 135 | 0.9195 +/- 0.0105 | 0.8814 +/- 0.0192 | +0.0380 | 10/0/0 | 0.007812 |
| Data | 204 | 0.9511 +/- 0.0019 | 0.8245 +/- 0.0188 | +0.1266 | 10/0/0 | 0.007812 |

The full raw per-seed CSVs are in `raw_per_seed/`.
