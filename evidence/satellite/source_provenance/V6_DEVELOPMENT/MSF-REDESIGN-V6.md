# MSF Satellite Health V6 — Final Design

V6 uses a 582-feature telemetry window representation (288 univariate + 294 cross-channel) and applies the original MSF-MOFS search locally, not globally. Four high-dimensional subsystems use MSF; communication and propulsion use full local features because the final experiments do not justify forcing MSF there.

## Deployed feature plan

- power: 18 / 267 (MSF)
- thermal: 11 / 135 (MSF)
- attitude: 15 / 135 (MSF)
- data: 10 / 204 (MSF)
- communication: 69 / 69 (direct)
- propulsion: 24 / 24 (direct)
- unique online feature union: 140 / 582

## Online health path

```text
telemetry frame -> 64-point causal history -> selected local features -> logistic evidence -> evidence smoothing -> 3-state DBN -> subsystem health -> overall health
```

The data-stream service remains responsible for point generation/transport and optional training truth. Ground truth is never required by online inference.
