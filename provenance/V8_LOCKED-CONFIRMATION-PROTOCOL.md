# Locked Confirmation Protocol V8

Status: frozen before confirmation outcomes are generated or inspected.

## Frozen method

The deployed/search method is the V6 subsystem-local design. No feature definition, subsystem mapping, MSF parameter, health threshold, evidence model, DBN parameter, or detection criterion is changed after this protocol is written.

Primary MSF implementation:
- one Fisher ranking on confirmation training data;
- mixed cardinality rule S(D)={10,0.05D,0.10D,0.20D}, inject the three smallest distinct top-Fisher subsets;
- population=12, generations=6;
- StandardScaler + 5-NN, stratified 3-fold wrapper;
- objectives: 1-G-mean and selected-feature ratio;
- standard NSGA-II downstream search unchanged after initialization.

Primary high-dimensional subsystems: power, thermal, attitude, data.
Boundary subsystem: communication. Propulsion remains direct because D=24.

## Confirmation data bank (predeclared)

A new synthetic scenario bank is generated with seeds disjoint from V6 development.

- generator seed base: 8,600,000;
- length: 240 ticks; stride: 8; causal maximum window: 64;
- normal scenarios per split: 20;
- univariate modes: drift, spike, step, noise, oscillation, intermittent;
- univariate base magnitudes are the frozen V6 values multiplied by a deterministic unseen factor in [0.88, 1.15];
- coupled pairs: p01_p02, p03_p04, p06_p12, p10_p11, p01_p08;
- coupled modes: invert, lag, decouple;
- coupled severities: 0.55, 0.85, 1.15 (V6 used 0.65 and 1.0);
- coupled lags: 6, 10, 14 ticks;
- fault onset: deterministic 68..104 ticks;
- nuisance noiseScale: deterministic range ~0.75..1.22;
- nuisance amplitudeScale: deterministic range ~0.78..1.18;
- train/validation/test remain scenario-disjoint;
- transition windows remain label=-1 and are excluded from Fisher/wrapper binary training.

## Search randomness and planned comparisons

Matched search seeds: 192..201 (not used in V6 search development).

RQ1 family (primary): MSF vs matched Random held-out Pareto HV across power/thermal/attitude/data.
RQ2 family (primary): MSF validation-selected subset vs Fisher Top-K at the exact same K on independent test data across the same four subsystems.
RQ3 family (mechanism): MSF vs Sparse-control and Sparse-control vs Random across the same four subsystems.
Communication is reported only as a boundary/exploratory result and is excluded from the four-subsystem Holm family.

Statistics: paired two-sided Wilcoxon; Holm correction within each four-comparison RQ family; W/T/L and paired rank-biserial effect.

## Frozen deployment confirmation

The V6 deployed subsystem-health model is evaluated without retraining on a disjoint scenario set.
Detection criterion remains frozen:
`target subsystem evidence >= 0.5 OR target subsystem health < 85 OR overall health < 85`.

Health confirmation uses new seeds, the same unseen nuisance ranges, unseen coupled severities 0.55/0.85/1.15, and deterministic unseen univariate magnitude factors. Ground truth is used only for scoring and is never passed to online inference.

A separate progressive-drift confirmation uses new seeds and a deterministic family of drift slopes; no result-driven threshold change is permitted.

## No-tuning rule

After generation starts, failed scenarios, non-significant tests, or weaker effect sizes are retained as confirmation outcomes. No parameter will be changed in response to confirmation results.
