// snn.mjs — a SPIKING NEURAL NET · leaky integrate-and-fire (LIF) neurons + spike-timing-dependent
// plasticity (STDP). An EVENT-TIME classifier: fed a spatiotemporal spike pattern, it learns which output
// neuron should fire. Its honest edge over a rate/deep model is COINCIDENCE DETECTION — spikes that arrive
// together drive a neuron over threshold; the SAME spikes spread out in time LEAK away and never fire. That
// is information in the TIMING, which a rate model throws away. Pure, zero-dep, Node + browser.
//
// HONEST SCOPE: this is a simulator, not neuromorphic hardware, and one leaky layer — it does event-time
// things (coincidence, order) that deep nets do awkwardly; it does NOT do what deep nets do well.

import classify from './attractor.mjs';   // the-spike → attractor: classify a neuron's own liveness regime

// LIF step: v leaks, integrates input current I, fires + resets at threshold.
export function lif(v, I, { leak = 0.82, thresh = 1.0, reset = 0 } = {}) {
  v = v * leak + I;
  if (v >= thresh) return { v: reset, spike: true };
  return { v, spike: false };
}

// STDP window: Δw as a function of (t_post − t_pre). Causal (pre before post, dt>0) → potentiate;
// acausal (post before pre, dt<0) → depress. The classic asymmetric exponential curve.
export function stdp(dt, { Aplus = 1, Aminus = 1, tau = 6 } = {}) {
  return dt >= 0 ? Aplus * Math.exp(-dt / tau) : -Aminus * Math.exp(dt / tau);
}

function rng(seed) { let s = seed >>> 0 || 1; return () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 4294967296; }; }

// a pattern = array of [inputIndex, time] spikes. Build per-timestep input lists.
function byTime(pattern, T) { const b = Array.from({ length: T }, () => []); for (const [i, t] of pattern) if (t >= 0 && t < T) b[t].push(i); return b; }

export function makeNet({ inputs, classes, seed = 1, thresh = 1.0, leak = 0.82, reset = 0 } = {}) {
  const r = rng(seed);
  const P = { thresh, leak, reset };
  const W = Array.from({ length: classes }, () => Array.from({ length: inputs }, () => 0.15 + 0.1 * r()));

  // run one pattern through the net; return each class's first-spike time, spike count, and the winner (earliest).
  function present(pattern, T) {
    const bt = byTime(pattern, T), v = new Array(classes).fill(0), first = new Array(classes).fill(Infinity), count = new Array(classes).fill(0);
    let winner = -1, winnerT = Infinity;
    for (let t = 0; t < T; t++) {
      for (let c = 0; c < classes; c++) {
        let I = 0; for (const i of bt[t]) I += W[c][i];
        const s = lif(v[c], I, P); v[c] = s.v;
        if (s.spike) { count[c]++; if (first[c] === Infinity) first[c] = t; if (t < winnerT) { winnerT = t; winner = c; } }
      }
    }
    return { first, count, winner };
  }
  const classify = (pattern, T) => present(pattern, T).winner;

  // reward-modulated STDP: reinforce the correct class along the causal (pre→post) window; if it did not
  // fire, potentiate it so it will; and depress whichever WRONG class fired first.
  function train(patterns, labels, { T, epochs = 80, lr = 0.02, wmax = 3 } = {}) {
    const hist = [];
    for (let e = 0; e < epochs; e++) {
      let correct = 0;
      for (let k = 0; k < patterns.length; k++) {
        const pat = patterns[k], L = labels[k], res = present(pat, T);
        if (res.winner === L) correct++;
        const tL = res.first[L];
        for (const [i, ti] of pat) {
          W[L][i] += lr * (tL === Infinity ? 0.6 : stdp(tL - ti));                                    // potentiate the target
          if (res.winner !== -1 && res.winner !== L) W[res.winner][i] -= lr * 0.6 * Math.exp(-Math.abs((res.first[res.winner] || 0) - ti) / 6);  // depress the wrong winner
        }
        for (let c = 0; c < classes; c++) for (let i = 0; i < inputs; i++) W[c][i] = Math.max(0, Math.min(wmax, W[c][i]));
      }
      hist.push(correct / patterns.length);
    }
    return hist;
  }

  const accuracy = (patterns, labels, T) => patterns.reduce((a, p, k) => a + (classify(p, T) === labels[k] ? 1 : 0), 0) / patterns.length;
  return { present, classify, train, accuracy, W, P };
}

// helper: make a random spatiotemporal pattern — `k` of `inputs` spike, each at a random time in [0,T).
export function randomPattern(inputs, k, T, rand) { const idx = [...Array(inputs).keys()]; for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1));[idx[i], idx[j]] = [idx[j], idx[i]]; } return idx.slice(0, k).map(i => [i, Math.floor(rand() * T)]); }
export { rng as _rng };

// ── THE ATTRACTOR WIRE (the-spike → attractor) ── run a neuron over a per-step input-current series → its
// membrane trajectory; attractor then says which dynamical regime it is in: a live ATTRACTOR (sustained
// firing), FLATLINE (dead / no drive), or ESCAPED. The spike organ can now report its OWN liveness.
export function membraneTrace(currents, params = {}) {
  const P = { leak: 0.82, thresh: 1.0, reset: 0, ...params };
  let v = 0; const trace = [];
  for (const I of (Array.isArray(currents) ? currents : [])) { const s = lif(v, I, P); v = s.v; trace.push(s.spike ? P.thresh : v); }
  return trace;
}
export function liveness(currents, params = {}) { return classify(membraneTrace(currents, params)); }

export default { lif, stdp, makeNet, randomPattern, membraneTrace, liveness };
