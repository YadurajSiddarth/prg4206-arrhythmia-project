// Verifies the JS TreeSHAP against a brute-force Shapley computation (the definition) on random trees.
const E = require("../docs/engine.js");
let seed = 12345;
const rnd = () => { seed = (seed * 1664525 + 1013904223) % 4294967296; return seed / 4294967296; };

function randomTree(M, maxDepth) {
  const T = { f: [], t: [], l: [], r: [], m: [], c: [], v: [] };
  function build(depth) {
    const id = T.f.length;
    T.f.push(-1); T.t.push(0); T.l.push(-1); T.r.push(-1); T.m.push(-1); T.c.push(0); T.v.push(0);
    if (depth >= maxDepth || (depth > 0 && rnd() < 0.25)) {
      T.v[id] = rnd() * 2 - 1; T.c[id] = 1 + Math.floor(rnd() * 50); return id;
    }
    T.f[id] = Math.floor(rnd() * M); T.t[id] = rnd();
    const l = build(depth + 1), r = build(depth + 1);
    T.l[id] = l; T.r[id] = r; T.m[id] = rnd() < 0.5 ? l : r; T.c[id] = T.c[l] + T.c[r];
    return id;
  }
  build(0);
  return T;
}
function bruteShap(model, x) {
  const M = model.features.length, phi = new Array(M).fill(0);
  const fact = (n) => (n <= 1 ? 1 : n * fact(n - 1));
  function ev(tree, node, S) {
    if (tree.f[node] < 0) return tree.v[node];
    const f = tree.f[node];
    if (S.has(f)) {
      const xv = x[f];
      const nxt = Number.isNaN(xv) ? tree.m[node] : (Math.fround(xv) < Math.fround(tree.t[node]) ? tree.l[node] : tree.r[node]);
      return ev(tree, nxt, S);
    }
    const l = tree.l[node], r = tree.r[node];
    return (tree.c[l] * ev(tree, l, S) + tree.c[r] * ev(tree, r, S)) / tree.c[node];
  }
  const v = (S) => model.trees.reduce((s, t) => s + ev(t, 0, S), model.offset);
  for (let i = 0; i < M; i++) {
    for (let mask = 0; mask < (1 << M); mask++) {
      if (mask & (1 << i)) continue;
      const S = new Set(); let size = 0;
      for (let j = 0; j < M; j++) if (mask & (1 << j)) { S.add(j); size++; }
      const w = (fact(size) * fact(M - size - 1)) / fact(M);
      const S2 = new Set(S); S2.add(i);
      phi[i] += w * (v(S2) - v(S));
    }
  }
  return phi;
}
let worst = 0, cases = 0;
for (let trial = 0; trial < 60; trial++) {
  const M = 3 + Math.floor(rnd() * 3);                       // 3-5 features -> repeated features on paths
  const model = { offset: rnd() - 0.5, features: Array.from({ length: M }, (_, i) => "f" + i),
                  trees: Array.from({ length: 1 + Math.floor(rnd() * 4) }, () => randomTree(M, 4)) };
  for (let k = 0; k < 5; k++) {
    const x = Array.from({ length: M }, () => (rnd() < 0.15 ? NaN : rnd()));
    const got = E.explain(model, x), ref = bruteShap(model, x);
    got.phi.forEach((p, i) => { worst = Math.max(worst, Math.abs(p - ref[i])); });
    const eff = got.phi.reduce((a, b) => a + b, 0) + got.bias - got.margin;     // efficiency property
    worst = Math.max(worst, Math.abs(eff));
    cases++;
  }
}
console.log(`cases: ${cases}, worst abs error vs brute-force Shapley: ${worst.toExponential(2)}`);
if (worst > 1e-9) { console.log("FAIL"); process.exit(1); }
console.log("PASS");
