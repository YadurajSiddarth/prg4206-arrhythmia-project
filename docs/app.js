(function () {
  "use strict";
  const M = window.ECG_MODEL, E = window.ECGEngine;
  const F = M.features;
  const $ = (id) => document.getElementById(id);

  // ---------- readable feature names ----------
  const DESC = {
    mean: "Mean amplitude (mV)", std: "Amplitude variability, std (mV)", min: "Minimum amplitude (mV)",
    max: "Maximum amplitude (mV)", range: "Peak-to-peak range (mV)", rms: "RMS amplitude (mV)",
    energy: "Signal energy (mV²)", skew: "Skewness", kurtosis: "Kurtosis (peakedness)",
    bp_baseline: "Relative power 0.5–5 Hz", bp_qrs: "Relative power 5–15 Hz (QRS band)",
    bp_high: "Relative power 15–40 Hz", dom_freq: "Dominant frequency (Hz)", spec_centroid: "Spectral centroid (Hz)",
    spec_entropy: "Spectral entropy (bits)", n_beats: "Detected R-peaks in 10 s", hr_mean: "Mean heart rate (bpm)",
    rr_mean: "Mean RR interval (s)", rr_std: "RR interval std, SDNN (s)", rmssd: "RMSSD of RR intervals (s)",
    pnn50: "pNN50 (fraction of RR changes > 50 ms)", rr_cv: "RR variability (coefficient of variation)",
    r_amp_mean: "Mean R-peak amplitude (mV)", r_amp_std: "R-peak amplitude std (mV)",
  };
  function describe(name) {
    const i = name.indexOf("_");
    const lead = name.slice(0, i), key = name.slice(i + 1);
    return { text: DESC[key] || key, lead: "lead " + lead };
  }
  const fmt = (v) => (v === null || Number.isNaN(v) ? "missing" : Number(v).toPrecision(3));
  const toVec = (arr) => arr.map((v) => (v === null || v === undefined ? NaN : v));

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }

  // ---------- tabs ----------
  const tabs = ["demo", "csv", "record", "manual"];
  function showTab(name) {
    tabs.forEach((t) => {
      $("tab-" + t).setAttribute("aria-selected", String(t === name));
      $("panel-" + t).hidden = t !== name;
    });
    $("error").hidden = true;
    if (name === "demo") runDemo();
    else if (name === "csv" && csv.rows.length) runCsvRow();
    else if (name === "record" && rec.last) render(rec.last.x, rec.last.note, rec.last.trace);
    else $("result").hidden = true;
  }
  tabs.forEach((t) => $("tab-" + t).addEventListener("click", () => showTab(t)));

  function showError(msg) { const e = $("error"); e.textContent = msg; e.hidden = false; }

  // ---------- demo ----------
  const sel = $("demo-select");
  M.demo.forEach((d, i) => {
    const o = document.createElement("option");
    o.value = i; o.textContent = `ECG ${d.id}`;
    sel.appendChild(o);
  });
  function runDemo() {
    const d = M.demo[Number(sel.value)];
    $("error").hidden = true;
    render(toVec(d.x), `Ground truth (shown for demo only): ${d.label}` + (d.label === "Abnormal" ? ` (${d.superclasses})` : ""));
  }
  sel.addEventListener("change", runDemo);

  // ---------- CSV (single row or a whole table such as data/features.csv) ----------
  const csv = { idx: {}, rows: [], visible: [], shown: [] };
  const splitLine = (l) => l.split(",").map((s) => s.trim().replace(/^"|"$/g, ""));

  function loadCsv(text) {
    const lines = text.replace(/\r/g, "").split("\n").filter((l) => l.trim() !== "");
    if (lines.length < 2) throw new Error("The CSV needs a header row and at least one data row.");
    const idx = {};
    splitLine(lines[0]).forEach((h, i) => { idx[h] = i; });
    const absent = F.filter((f) => !(f in idx));
    if (absent.length) throw new Error(`Missing ${absent.length} feature column(s), e.g. ${absent.slice(0, 3).join(", ")}.`);
    csv.idx = idx;
    csv.rows = lines.slice(1).map(splitLine);
  }
  function rowVector(raw) {
    return F.map((f) => {
      const v = raw[csv.idx[f]];
      if (v === undefined || v === "" || v.toLowerCase() === "nan") return NaN;
      const n = Number(v);
      if (Number.isNaN(n)) throw new Error(`Column ${f} is not a number: "${v}".`);
      return n;
    });
  }
  function rowInfo(raw) {
    const g = (c) => (c in csv.idx ? raw[csv.idx[c]] : undefined);
    let truth = g("true_label");
    if (truth === undefined && g("abnormal") !== undefined && g("abnormal") !== "") truth = Number(g("abnormal")) === 1 ? "Abnormal" : "Normal";
    return { id: g("ecg_id"), split: g("split"), truth, sup: g("superclasses") };
  }
  function rebuildRowSelect(selected) {
    const sel = $("csv-row"); sel.replaceChildren();
    csv.shown.forEach((i) => {
      const inf = rowInfo(csv.rows[i]), parts = [`Row ${i + 1}`];
      if (inf.id) parts.push(`ECG ${inf.id}`);
      if (inf.split) parts.push(inf.split);
      if (inf.truth) parts.push(inf.truth);
      sel.appendChild(new Option(parts.join(" · "), String(i)));
    });
    if (selected !== undefined) sel.value = String(selected);
    $("csv-info").textContent = `${csv.visible.length} matching record(s)` +
      (csv.visible.length > csv.shown.length ? `; the list shows ${csv.shown.length} of them – use “Random record” to sample from all.` : ".");
  }
  function applyFilter() {
    const sp = $("csv-split").value, hasSplit = "split" in csv.idx;
    csv.visible = csv.rows.map((_, i) => i).filter((i) => !hasSplit || sp === "all" || csv.rows[i][csv.idx.split] === sp);
    csv.shown = csv.visible.slice(0, 1000);
    rebuildRowSelect();
    runCsvRow();
  }
  function runCsvRow() {
    const sel = $("csv-row");
    if (!sel.value) { $("result").hidden = true; return showError("No records match this filter."); }
    $("error").hidden = true;
    const raw = csv.rows[Number(sel.value)], inf = rowInfo(raw);
    let note = inf.truth ? `Ground truth from the file: ${inf.truth}` + (inf.truth === "Abnormal" && inf.sup ? ` (${inf.sup.replace(/\|/g, ", ")})` : "") : "Source: uploaded CSV";
    if (inf.split === "train" || inf.split === "val") note += ` · this record was used for ${inf.split === "train" ? "training" : "tuning"}, so its prediction is not an independent test`;
    try { render(rowVector(raw), note); } catch (e) { $("result").hidden = true; showError(e.message); }
  }
  $("csv-file").addEventListener("change", (ev) => {
    const file = ev.target.files[0];
    if (!file) return;
    file.text().then((t) => {
      loadCsv(t);
      const s = $("csv-split"); s.replaceChildren();
      const splits = "split" in csv.idx ? [...new Set(csv.rows.map((r) => r[csv.idx.split]))] : [];
      ["all", ...splits].forEach((v) => s.appendChild(new Option(v === "all" ? "all records" : v, v)));
      s.value = splits.includes("test") ? "test" : "all";       // default to held-out test records
      s.parentElement.hidden = !splits.length;
      $("csv-picker").hidden = false;
      applyFilter();
    }).catch((e) => { csv.rows = []; $("csv-picker").hidden = true; $("result").hidden = true; showError(e.message); });
  });
  $("csv-split").addEventListener("change", applyFilter);
  $("csv-row").addEventListener("change", runCsvRow);
  $("csv-random").addEventListener("click", () => {
    if (!csv.visible.length) return;
    const pick = csv.visible[Math.floor(Math.random() * csv.visible.length)];
    if (!csv.shown.includes(pick)) csv.shown = [pick, ...csv.shown.slice(0, 999)];
    rebuildRowSelect(pick);
    runCsvRow();
  });

  // ---------- raw ECG record (.hea + .dat) ----------
  const rec = { last: null };
  function drawTrace(tr) {
    const wrap = $("ecg-plot-wrap"), svg = $("ecg-plot");
    wrap.hidden = !tr;
    if (!tr) return;
    svg.replaceChildren();
    const W = 800, H = 220, pad = 14, n = tr.filtered.length, ns = "http://www.w3.org/2000/svg";
    let lo = Infinity, hi = -Infinity;
    [tr.raw, tr.filtered].forEach((arr) => arr.forEach((v) => { if (v < lo) lo = v; if (v > hi) hi = v; }));
    const px = (i) => pad + ((W - 2 * pad) * i) / (n - 1), py = (v) => H - pad - ((H - 2 * pad) * (v - lo)) / (hi - lo || 1);
    [["raw", tr.raw], ["filt", tr.filtered]].forEach(([cls, arr]) => {
      const p = document.createElementNS(ns, "path");
      p.setAttribute("class", cls);
      p.setAttribute("d", arr.map((v, i) => (i ? "L" : "M") + px(i).toFixed(1) + " " + py(v).toFixed(1)).join(" "));
      svg.appendChild(p);
    });
    tr.peaks.forEach((i) => {
      const c = document.createElementNS(ns, "circle");
      c.setAttribute("class", "peak"); c.setAttribute("r", "3.5");
      c.setAttribute("cx", px(i).toFixed(1)); c.setAttribute("cy", py(tr.filtered[i]).toFixed(1));
      svg.appendChild(c);
    });
  }
  $("rec-files").addEventListener("change", async (ev) => {
    try {
      const files = Array.from(ev.target.files);
      const hea = files.find((f) => /\.hea$/i.test(f.name)), dat = files.find((f) => /\.dat$/i.test(f.name));
      if (!hea || !dat) throw new Error("Please select both the .hea and the .dat file of one record.");
      const base = (n) => n.replace(/\.(hea|dat)$/i, "");
      if (base(hea.name) !== base(dat.name)) throw new Error("The .hea and .dat files must belong to the same record (same file name).");
      const r = ECGExtract.readWfdb(await hea.text(), await dat.arrayBuffer());
      const { feats, trace } = ECGExtract.extractFeatures(r, M.meta.leads, ["II"]);
      const absent = F.filter((f) => !(f in feats));
      if (absent.length) throw new Error(`The extractor did not produce: ${absent.slice(0, 3).join(", ")}.`);
      rec.last = { x: F.map((f) => feats[f]), note: `Source: record ${base(hea.name)} · features extracted in your browser`, trace: trace.II || null };
      $("error").hidden = true;
      render(rec.last.x, rec.last.note, rec.last.trace);
    } catch (e) { rec.last = null; $("result").hidden = true; showError(e.message); }
  });

  // ---------- manual form ----------
  const grid = $("manual-grid");
  F.forEach((f, i) => {
    const d = describe(f), wrap = el("div");
    const lab = el("label", null, `${d.text} · ${d.lead}`);
    lab.htmlFor = "in-" + i;
    const inp = document.createElement("input");
    inp.type = "number"; inp.step = "any"; inp.id = "in-" + i;
    inp.placeholder = String(M.stats[f].median);
    inp.title = `Typical range in training data: ${M.stats[f].p1} to ${M.stats[f].p99}`;
    wrap.append(lab, inp); grid.appendChild(wrap);
  });
  const inputs = () => F.map((_, i) => $("in-" + i));
  $("fill-typical").addEventListener("click", () => inputs().forEach((inp, i) => { inp.value = M.stats[F[i]].median; }));
  $("clear-form").addEventListener("click", () => inputs().forEach((inp) => { inp.value = ""; }));
  $("predict-manual").addEventListener("click", () => {
    const x = inputs().map((inp) => (inp.value.trim() === "" ? NaN : Number(inp.value)));
    if (x.every(Number.isNaN)) { $("result").hidden = true; return showError("Enter at least one value (or use “Fill with typical values”)."); }
    $("error").hidden = true;
    render(x, "Source: manual entry");
  });

  // ---------- result ----------
  function render(x, truthText, trace) {
    const r = E.explain(M, x);
    const abnormal = r.prob >= 0.5;
    $("result").hidden = false;
    const badge = $("badge");
    badge.textContent = abnormal ? "ABNORMAL" : "NORMAL";
    badge.className = "badge " + (abnormal ? "abn" : "nrm");
    $("prob").textContent = (r.prob * 100).toFixed(1) + "%";
    $("conf").textContent = (Math.max(r.prob, 1 - r.prob) * 100).toFixed(1) + "%";
    $("meter-fill").style.left = (r.prob * 100).toFixed(1) + "%";
    drawTrace(trace);
    const truth = $("truth");
    truth.textContent = truthText || ""; truth.hidden = !truthText;
    const nMissing = x.filter((v) => Number.isNaN(v)).length;
    const mn = $("missing-note");
    mn.textContent = nMissing ? `${nMissing} of ${F.length} features were missing; the model's default branch was used for them.` : "";
    mn.hidden = !nMissing;

    // top-10 contributions
    const order = F.map((_, i) => i).sort((a, b) => Math.abs(r.phi[b]) - Math.abs(r.phi[a]));
    const top = order.slice(0, 10), maxAbs = Math.abs(r.phi[top[0]]) || 1;
    const bars = $("bars"); bars.replaceChildren();
    top.forEach((i) => {
      const d = describe(F[i]), row = el("div", "bar-row");
      const name = el("div", "bar-name", d.text);
      name.appendChild(el("small", null, `${d.lead} · value ${fmt(x[i])}`));
      const area = el("div", "bar-area"), bar = el("div", "bar " + (r.phi[i] >= 0 ? "abn" : "nrm"));
      bar.style.width = (Math.abs(r.phi[i]) / maxAbs * 50).toFixed(1) + "%";
      area.appendChild(bar);
      row.append(name, area, el("div", "bar-val", (r.phi[i] >= 0 ? "+" : "") + r.phi[i].toFixed(3)));
      bars.appendChild(row);
    });

    // all-features table
    const t = $("all-features"); t.replaceChildren();
    const head = el("tr"); ["Feature", "Lead", "Value", "SHAP (log-odds)"].forEach((h, k) => head.appendChild(el("th", k > 1 ? "num" : "", h)));
    t.appendChild(head);
    order.forEach((i) => {
      const d = describe(F[i]), tr = el("tr");
      tr.append(el("td", null, d.text), el("td", null, d.lead.replace("lead ", "")),
        el("td", "num", fmt(x[i])), el("td", "num", (r.phi[i] >= 0 ? "+" : "") + r.phi[i].toFixed(3)));
      t.appendChild(tr);
    });
    const foot = el("tr");
    foot.append(el("td", null, "Base value (average model output, log-odds)"), el("td"), el("td"), el("td", "num", r.bias.toFixed(3)));
    t.appendChild(foot);
  }

  // ---------- about (text is generated from the exported model so it always matches the numbers) ----------
  const m = M.meta, xgbRow = M.metrics.find((r) => r.Model === "XGBoost");
  const n = (v) => Number(v).toLocaleString("en-US");
  const leadTxt = m.leads.length === 1 ? `Lead ${m.leads[0]}` : `leads ${m.leads.join(", ")}`;
  let about = `XGBoost classifier trained on engineered ${leadTxt} ECG features from ${m.dataset}. ` +
    `The experiment used ${n(m.n_train)} training records, ${n(m.n_val)} validation records and ${n(m.n_test)} held-out test records, ` +
    `with ${n(m.n_train + m.n_val + m.n_test)} usable records in total. ` +
    `The model uses ${F.length} statistical, temporal and frequency-domain features. ` +
    `Patient records are kept within their designated data split. ` +
    `Metrics are from the held-out test set at a 0.5 classification threshold. XGBoost is the selected model.`;
  if (xgbRow && xgbRow["AUC 95% CI"]) about += ` ROC-AUC 95% bootstrap confidence interval: ${String(xgbRow["AUC 95% CI"]).replace("-", "–")}.`;
  $("about").textContent = about;

  const limits = [
    "Binary classification: NORM versus abnormal ECGs.",
    `Uses engineered features from ${m.leads.length === 1 ? leadTxt + " only" : leadTxt}.`,
    "No external-dataset or clinical validation.",
  ];
  if (xgbRow && typeof xgbRow["Recall (sensitivity)"] === "number") {
    limits.push(`About ${((1 - xgbRow["Recall (sensitivity)"]) * 100).toFixed(1)}% of abnormal test records were missed at the selected threshold.`);
  }
  limits.push("SHAP explains model behaviour, not medical correctness.");
  limits.forEach((t) => $("limits").appendChild(el("li", null, t)));

  if (M.metrics.length) {
    const wanted = ["Model", "Accuracy", "Precision", "Recall (sensitivity)", "Specificity", "F1", "ROC-AUC", "AUC 95% CI", "PR-AUC"];
    const cols = wanted.filter((c) => c in M.metrics[0]);                       // show only the columns the export contains
    const t = $("metrics"), head = el("tr");
    cols.forEach((c, k) => head.appendChild(el("th", k ? "num" : "", c)));
    t.appendChild(head);
    M.metrics.forEach((row) => {
      const tr = el("tr", row.Model === "XGBoost" ? "hl" : "");
      cols.forEach((c, k) => {
        const v = row[c], text = typeof v === "number" ? v.toFixed(4) : String(v);   // numbers to 4 decimals, the CI stays text
        tr.appendChild(el("td", k ? "num" : "", text));
      });
      t.appendChild(tr);
    });
  }

  // ---------- self-test: the page re-checks itself against XGBoost's own SHAP output ----------
  (function selfTest() {
    const pill = $("selftest");
    try {
      let worst = 0;
      M.selftest.forEach((s) => {
        const r = E.explain(M, toVec(s.x));
        worst = Math.max(worst, Math.abs(r.margin - s.margin), Math.abs(r.bias - s.bias));
        r.phi.forEach((p, i) => { worst = Math.max(worst, Math.abs(p - s.phi[i])); });
      });
      const ok = worst < 1e-3;
      pill.textContent = ok ? `Model self-test passed (max deviation ${worst.toExponential(1)})` : `Model self-test FAILED (${worst.toExponential(1)})`;
      pill.className = "pill " + (ok ? "ok" : "bad");
    } catch (e) { pill.textContent = "Model self-test error: " + e.message; pill.className = "pill bad"; }
  })();

  showTab("demo");
})();
