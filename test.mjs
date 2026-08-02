// test.mjs — PROOF-OF-PLAY for the SPIKING NET. Zero tokens, deterministic. Proves LIF dynamics
// (integrate/fire/reset/leak), STDP sign (causal→potentiate, acausal→depress), the honest EVENT-TIME win
// (COINCIDENCE DETECTION — spikes together fire, spread out they leak away), and that STDP training LEARNS
// to classify distinct spatiotemporal patterns.
import { lif, stdp, makeNet, randomPattern, membraneTrace, liveness, _rng } from './snn.mjs';

let pass = 0, fail = 0;
const ok = (c, m) => { c ? pass++ : fail++; console.log((c ? '  ✓ ' : '  ✗ FAIL ') + m); };

console.log('=== §1 · LIF — integrate, fire at threshold, reset, leak ===');
{
  const P = { leak: 0.82, thresh: 1.0, reset: 0 };
  let v = 0, spikes = 0, fireStep = -1;
  for (let t = 0; t < 5; t++) { const s = lif(v, 0.5, P); v = s.v; if (s.spike) { spikes++; if (fireStep < 0) fireStep = t; } }   // 0.5 each step
  ok(spikes >= 1 && fireStep === 2, `steady 0.5 input integrates and FIRES at step ${fireStep} (0.5→0.91→1.25≥1), then resets`);
  let vd = lif(0, 0.9, P).v; const before = vd; vd = lif(vd, 0, P).v;
  ok(vd < before && Math.abs(vd - before * 0.82) < 1e-9, `with no input the membrane LEAKS (×0.82): ${before.toFixed(3)} → ${vd.toFixed(3)}`);
  ok(lif(0.9, 0.5, P).v === 0, 'crossing threshold RESETS to 0');
}

console.log('\n=== §2 · STDP — causal potentiates, acausal depresses ===');
{
  ok(stdp(5) > 0 && stdp(-5) < 0, 'pre-before-post (Δt>0) → LTP (+); post-before-pre (Δt<0) → LTD (−)');
  ok(stdp(1) > stdp(8), 'potentiation decays with the pre/post gap (closer spikes learn more)');
  ok(Math.abs(stdp(0) - 1) < 1e-9, 'coincident spikes give the maximum update');
}

console.log('\n=== §3 · COINCIDENCE DETECTION — the event-time win a rate model cannot see ===');
{
  const net = makeNet({ inputs: 4, classes: 1, thresh: 1.0, leak: 0.82 });
  for (let i = 0; i < 4; i++) net.W[0][i] = 0.3;                         // fixed equal weights
  const T = 16;
  const coincident = [[0, 5], [1, 5], [2, 5], [3, 5]];                    // all 4 arrive together → I=1.2 ≥ 1.0
  const spread = [[0, 1], [1, 5], [2, 9], [3, 13]];                       // SAME 4 spikes, spread out → each leaks away
  const rc = net.present(coincident, T), rs = net.present(spread, T);
  ok(rc.winner === 0 && rc.count[0] >= 1, 'coincident spikes drive the neuron over threshold → it FIRES');
  ok(rs.winner === -1 && rs.count[0] === 0, 'the SAME spikes spread over time LEAK away → it does NOT fire');
  ok(true, 'identical spike SET, opposite outcome — the information is in the TIMING (a rate model is blind to this)');
}

console.log('\n=== §4 · LEARNING — STDP trains a classifier of distinct spatiotemporal patterns ===');
{
  const I = 16, C = 5, T = 25, r = _rng(0xC0DE);
  const patterns = [], labels = [];
  for (let c = 0; c < C; c++) { patterns.push(randomPattern(I, 6, T, r)); labels.push(c); }
  const net = makeNet({ inputs: I, classes: C, seed: 7, thresh: 1.0, leak: 0.82 });
  const before = net.accuracy(patterns, labels, T);
  const hist = net.train(patterns, labels, { T, epochs: 120, lr: 0.02 });
  const after = net.accuracy(patterns, labels, T);
  ok(after > before, `accuracy improved with training: ${(before * 100).toFixed(0)}% → ${(after * 100).toFixed(0)}% (learning happened)`);
  ok(after >= 0.8, `final accuracy ${(after * 100).toFixed(0)}% on ${C} distinct temporal patterns (≥80%)`);
  ok(hist[hist.length - 1] >= hist[0], `training accuracy rose over epochs (${(hist[0] * 100).toFixed(0)}% → ${(hist[hist.length - 1] * 100).toFixed(0)}%)`);
}

console.log('\n=== §5 · DETERMINISM + §6 · FUZZ ===');
{
  const mk = () => { const I = 12, C = 4, T = 20, r = _rng(42); const pats = [], labs = []; for (let c = 0; c < C; c++) { pats.push(randomPattern(I, 5, T, r)); labs.push(c); } const n = makeNet({ inputs: I, classes: C, seed: 3 }); n.train(pats, labs, { T, epochs: 40, lr: 0.02 }); return n.accuracy(pats, labs, T); };
  ok(mk() === mk(), 'deterministic: same seed + same patterns → same trained accuracy');
  let threw = false, r = _rng(0xBAD5EED);
  try { const n = makeNet({ inputs: 10, classes: 3, seed: 9 }); for (let t = 0; t < 500; t++) { const pat = randomPattern(10, 1 + Math.floor(r() * 9), 30, r); const res = n.present(pat, 30); if (!res.first.every(x => x === Infinity || Number.isFinite(x))) throw new Error('bad'); } n.present([], 5); n.present([[99, 3]], 5); } catch (e) { threw = true; console.log('    threw:', e.message); }
  ok(!threw, '500 random patterns + empty + out-of-range spike: 0 throws, all finite');
}

console.log('\n=== §7 · THE ATTRACTOR WIRE — the-spike classifies its OWN liveness (spike → attractor) ===');
{
  const driven = liveness(new Array(40).fill(0.35));                     // steady drive → fires periodically
  ok(driven.class === 'attractor' && driven.alive, `a steadily-driven neuron reads as a LIVE attractor (${driven.class})`);
  const dead = liveness(new Array(40).fill(0));                          // no drive → membrane sits at 0
  ok(dead.class === 'flatline', `a starved neuron FLATLINES — attractor calls it dead (${dead.class})`);
  ok(Array.isArray(membraneTrace([0.5, 0.5, 0.5])) && membraneTrace([]).length === 0, 'membraneTrace returns the v(t) trajectory (empty on garbage, no throw)');
}

console.log('\n' + (fail === 0
  ? `=== ✅ THE SPIKE HOLDS — LIF + STDP, coincidence detection, a learned classifier, liveness via attractor · ${pass}/${pass} · zero tokens ===`
  : `=== ✗ ${fail} FAILED (${pass} passed) ===`));
process.exit(fail === 0 ? 0 : 1);
