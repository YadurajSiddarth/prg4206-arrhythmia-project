/* Browser-side XGBoost inference + exact TreeSHAP explanations (no dependencies).
 *
 * A tree is stored as parallel arrays (node 0 = root):
 *   f: split feature index (-1 for a leaf)   t: split threshold (go LEFT if x < t)
 *   l/r: left/right child index   m: child taken when the feature is missing
 *   c: cover (training weight reaching the node)   v: leaf value (margin / log-odds)
 * margin(x) = offset + sum over trees of leaf value;  probability = sigmoid(margin)
 */
(function (root) {
  "use strict";

  const isMissing = (v) => v === null || v === undefined || Number.isNaN(v);

  function hotChild(tree, node, x) {
    const xv = x[tree.f[node]];
    if (isMissing(xv)) return tree.m[node];
    // XGBoost compares in float32
    return Math.fround(xv) < Math.fround(tree.t[node]) ? tree.l[node] : tree.r[node];
  }

  function treeLeafValue(tree, x) {
    let n = 0;
    while (tree.f[n] >= 0) n = hotChild(tree, n, x);
    return tree.v[n];
  }

  function predictMargin(model, x) {
    let s = model.offset;
    for (const tree of model.trees) s += treeLeafValue(tree, x);
    return s;
  }

  const sigmoid = (z) => 1 / (1 + Math.exp(-z));

  function treeExpectedValue(tree) {            // cover-weighted mean of the leaves
    let s = 0;
    for (let i = 0; i < tree.f.length; i++) if (tree.f[i] < 0) s += tree.v[i] * tree.c[i];
    return s / tree.c[0];
  }

  /* ---- TreeSHAP (Lundberg et al., 2018), path-dependent algorithm ---- */
  function treeShap(tree, x, phi) {
    function extend(path, pz, po, pi) {
      const ud = path.length;
      path.push({ d: pi, z: pz, o: po, w: ud === 0 ? 1 : 0 });
      for (let i = ud - 1; i >= 0; i--) {
        path[i + 1].w += (po * path[i].w * (i + 1)) / (ud + 1);
        path[i].w = (pz * path[i].w * (ud - i)) / (ud + 1);
      }
    }
    function unwind(path, idx) {
      const ud = path.length - 1;
      const one = path[idx].o, zero = path[idx].z;
      let next = path[ud].w;
      for (let i = ud - 1; i >= 0; i--) {
        if (one !== 0) {
          const tmp = path[i].w;
          path[i].w = (next * (ud + 1)) / ((i + 1) * one);
          next = tmp - (path[i].w * zero * (ud - i)) / (ud + 1);
        } else {
          path[i].w = (path[i].w * (ud + 1)) / (zero * (ud - i));
        }
      }
      for (let i = idx; i < ud; i++) {
        path[i].d = path[i + 1].d; path[i].z = path[i + 1].z; path[i].o = path[i + 1].o;
      }
      path.pop();
    }
    function unwoundSum(path, idx) {
      const ud = path.length - 1;
      const one = path[idx].o, zero = path[idx].z;
      let next = path[ud].w, total = 0;
      for (let i = ud - 1; i >= 0; i--) {
        if (one !== 0) {
          const tmp = (next * (ud + 1)) / ((i + 1) * one);
          total += tmp;
          next = path[i].w - (tmp * zero * (ud - i)) / (ud + 1);
        } else {
          total += path[i].w / zero / ((ud - i) / (ud + 1));
        }
      }
      return total;
    }
    function recurse(node, parentPath, pz, po, pi) {
      const path = parentPath.map((e) => ({ d: e.d, z: e.z, o: e.o, w: e.w }));
      extend(path, pz, po, pi);
      if (tree.f[node] < 0) {
        for (let i = 1; i < path.length; i++) {
          const w = unwoundSum(path, i);
          phi[path[i].d] += w * (path[i].o - path[i].z) * tree.v[node];
        }
        return;
      }
      const feat = tree.f[node];
      const hot = hotChild(tree, node, x);
      const cold = hot === tree.l[node] ? tree.r[node] : tree.l[node];
      let iz = 1, io = 1, k = -1;
      for (let i = 1; i < path.length; i++) if (path[i].d === feat) { k = i; break; }
      if (k >= 0) { iz = path[k].z; io = path[k].o; unwind(path, k); }
      const cv = tree.c[node];
      recurse(hot, path, (iz * tree.c[hot]) / cv, io, feat);
      recurse(cold, path, (iz * tree.c[cold]) / cv, 0, feat);
    }
    recurse(0, [], 1, 1, -1);
  }

  /* Returns { phi: [per-feature contribution in log-odds], bias, margin, prob } */
  function explain(model, x) {
    const phi = new Array(model.features.length).fill(0);
    let bias = model.offset;
    for (const tree of model.trees) {
      treeShap(tree, x, phi);
      bias += treeExpectedValue(tree);
    }
    const margin = predictMargin(model, x);
    return { phi, bias, margin, prob: sigmoid(margin) };
  }

  const api = { predictMargin, sigmoid, explain, treeShap, treeExpectedValue };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ECGEngine = api;
})(typeof self !== "undefined" ? self : this);
