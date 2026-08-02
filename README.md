# The Spike — a spiking neural net that reads timing

**▶ Live: https://sjgant80-hub.github.io/the-spike/**

Leaky integrate-and-fire (LIF) neurons + spike-timing-dependent plasticity (STDP). A neuron sums incoming
spikes but **leaks** between them — so spikes that arrive **together** push it over threshold and it
**fires**, while the **same spikes spread out** leak away and it stays quiet. That is information in the
**timing**, which a rate/deep model throws away. This is organ #5 (the last) of the anatomy cascade.

It's the **event-time** organ: it retrofits **attractor** (the estate's liveness classifier) with spikes,
and is the right shape for **niceassos' envelope-arrival streams** — event-time signals a deep net handles
awkwardly.

## Honest scope

A **simulator**, not neuromorphic hardware, and **one leaky layer**. It does event-time things — coincidence
detection, timing — that deep nets do awkwardly; it does **not** do what deep nets do well. No overclaim.

## Proven — `node test.mjs`, zero tokens, 14/14

- **LIF:** integrates a steady input and fires at threshold (0.5 → 0.91 → 1.25 ≥ 1), resets, and leaks (×0.82)
  with no input.
- **STDP:** causal (pre-before-post) potentiates, acausal depresses, magnitude decays with the gap, maximal
  at coincidence.
- **Coincidence detection (the win):** the *same* four spikes fire the neuron when coincident and do **not**
  when spread out — identical spike set, opposite outcome, because the info is in the timing.
- **Learning:** STDP training takes accuracy **0% → 100%** on 5 distinct spatiotemporal patterns.
- Deterministic; 500-pattern fuzz (incl. empty + out-of-range spikes), all finite, 0 throws.

## Files

`snn.mjs` (the kernel — `lif`, `stdp`, `makeNet` with present/classify/train/accuracy, `randomPattern`) ·
`test.mjs` (the 14/14 gate) · `index.html` (the live PWA — a membrane-potential trace with a coincidence
slider, and a "train (STDP)" demo that climbs 0% → 100% on temporal patterns) · `sw.js` +
`manifest.webmanifest` (offline). Zero-dep, Node + browser.

```bash
node test.mjs                 # the proof
python -m http.server 8080    # then open http://localhost:8080
```
