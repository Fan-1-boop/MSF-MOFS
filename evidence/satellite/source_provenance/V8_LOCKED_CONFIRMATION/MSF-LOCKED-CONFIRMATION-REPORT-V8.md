# MSF Satellite Health V8 — Locked Confirmation Report

## 1. Confirmation status

V8 is a post-freeze confirmation run. The confirmation protocol and hashes of the V6 method/model were written before the confirmation bank was generated and before search/health outcomes were inspected.

- New generator seed base: 8,600,000.
- 411 scenario-disjoint train/validation/test scenarios, 9,042 window samples, 582 candidate features.
- Search seeds: 192–201, disjoint from V6 development seeds 92–101.
- Primary family fixed to power, thermal, attitude, and data. Communication remains a boundary subsystem; propulsion remains direct because D=24.
- No parameter, threshold, feature definition, subsystem mapping, or deployment model was changed after confirmation outcomes appeared.

## 2. RQ1 — MSF vs matched Random

|Subsystem|D|MSF HV|Random HV|ΔHV|W/T/L|Holm p|RBC|
|---|---:|---:|---:|---:|---:|---:|---:|
|power|267|0.8054±0.0069|0.7222±0.0120|+0.0832|10/0/0|**0.007812**|1.000|
|thermal|135|0.8807±0.0161|0.8343±0.0200|+0.0464|10/0/0|**0.007812**|1.000|
|attitude|135|0.9195±0.0105|0.8814±0.0192|+0.0380|10/0/0|**0.007812**|1.000|
|data|204|0.9511±0.0019|0.8245±0.0188|+0.1266|10/0/0|**0.007812**|1.000|

**Confirmation conclusion:** all four primary subsystems remain 10/10 MSF > Random. Holm-corrected p=0.007812 for every comparison.

Communication remains exploratory/boundary: MSF HV=0.7430, Random=0.7174, W/T/L=8/0/2, raw p=0.048828. This does not change the frozen direct-feature deployment decision.

## 3. RQ2 — MSF vs Fisher Top-K at the same K

|Subsystem|Mean K|MSF test G|Fisher Top-K G|ΔG|W/L|Holm p|
|---|---:|---:|---:|---:|---:|---:|
|power|22.6|0.8314|0.8100|+0.0213|9/1|**0.039062**|
|thermal|13.9|0.9279|0.9123|+0.0156|7/3|0.083984|
|attitude|13.6|0.9680|0.8875|+0.0805|10/0|**0.007812**|
|data|11.7|0.9996|0.9301|+0.0695|10/0|**0.007812**|

**Confirmation conclusion:** the same-K wrapper advantage is confirmed for power, attitude, and data. Thermal remains positive in mean but does not pass Holm correction (7/10 wins, p=0.083984).

## 4. RQ3 — Mechanism attribution

|Subsystem|MSF HV|Sparse HV|Random HV|MSF–Sparse Holm p|Sparse–Random Holm p|
|---|---:|---:|---:|---:|---:|
|power|0.8054|0.7904|0.7222|0.058594|**0.007812**|
|thermal|0.8807|0.8866|0.8343|0.464844|**0.007812**|
|attitude|0.9195|0.9243|0.8814|0.464844|**0.007812**|
|data|0.9511|0.9004|0.8245|0.007812|**0.007812**|

**Confirmation conclusion:** Sparse-control > Random is Holm-significant in all four primary subsystems, confirming scale placement as the more stable mechanism. Additional Fisher-identity gain is clearly significant only for data; power is close but not Holm-significant.

## 5. Frozen V6 deployment on disjoint V8 scenarios

- Normal: 2/30 false-alarm scenarios; mean health=99.72, minimum=63.
- Univariate faults: **72/72 = 100.00%**; mean latency=9.21 tick.
- Coupled faults: **35/45 = 77.78%**; mean detected latency=35.71 tick.
- Progressive drift: **12/12** early warnings before limit crossing; mean warning latency=44.42 tick; mean lead=102.42 tick.

The frozen deployment remains strong on unseen single-channel faults and progressive drift, but coupled-fault robustness weakens under the wider confirmation bank and two nuisance-only normal scenarios false-alarm. These failures were retained without tuning.

Normal false alarms: v8-confirm-normal-13, v8-confirm-normal-17.

Missed coupled scenarios:
- `p01_p02` / `lag`; max evidence=0.021, min target health=100, min overall health=100.
- `p03_p04` / `lag`; max evidence=0.093, min target health=99, min overall health=100.
- `p10_p11` / `invert`; max evidence=0.272, min target health=91, min overall health=97.
- `p10_p11` / `lag`; max evidence=0.005, min target health=100, min overall health=100.
- `p10_p11` / `decouple`; max evidence=0.280, min target health=88, min overall health=97.
- `p01_p08` / `invert`; max evidence=0.100, min target health=99, min overall health=100.
- `p01_p08` / `lag`; max evidence=0.000, min target health=100, min overall health=100.
- `p01_p08` / `lag`; max evidence=0.160, min target health=99, min overall health=99.
- `p01_p08` / `decouple`; max evidence=0.007, min target health=100, min overall health=88.
- `p01_p08` / `decouple`; max evidence=0.078, min target health=100, min overall health=99.

## 6. Final interpretation

- **RQ1 is strongly confirmed.** This is the cleanest paper result.
- **RQ2 is partially confirmed.** Power, attitude, and data pass Holm; thermal does not.
- **RQ3 supports a conservative mechanism claim.** Scale placement is robust; Fisher identity is subsystem-dependent.
- **Deployment confirmation is credible rather than perfect.** Univariate detection and early warning remain strong, while coupled lag/decouple failures and nuisance false alarms define the current boundary.

All V8 results remain controlled synthetic telemetry evidence and must not be described as real-orbit accuracy.