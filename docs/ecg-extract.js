/* Browser-side port of ecg_features.py + a minimal WFDB (.hea / .dat) reader.
 *
 * It reproduces the Python pipeline: 0.5-40 Hz Butterworth band-pass (zero-phase, like scipy.signal.sosfiltfilt),
 * statistics, Welch band powers and R-peak / heart-rate-variability features. tests/test_extract.py checks that
 * the numbers match the Python code.
 *
 * Limits: PTB-XL 100 Hz records (the "_lr" files), WFDB signal format 16, at least 256 samples per lead.
 */
(function (root) {
  "use strict";
  const FS = 100;

  // scipy.signal.butter(3, [0.5, 40], 'band', fs=100, output='sos') and scipy.signal.sosfilt_zi(sos)
  const SOS = [
    [0.5103784499368984, 1.0207568998737968, 0.5103784499368984, 1.0, 1.250776646416673, 0.5477135943716132],
    [1.0, 0.0, -1.0, 1.0, -0.46305461775593415, -0.4898949450224769],
    [1.0, -2.0, 1.0, 1.0, -1.9682561479438663, 0.9692327528594086],
  ];
  const ZI = [[0.2191269705188594, 0.11081841398550034], [-0.7295054204557581, -0.7295054204557578], [0, 0]];

  const BANDS = { baseline: [0.5, 5], qrs: [5, 15], high: [15, 40] };

  /* ---------------- WFDB reader ---------------- */
  function parseHeader(text) {
    const lines = text.replace(/\r/g, "").split("\n").map((l) => l.trim()).filter((l) => l && !l.startsWith("#"));
    const first = lines[0].split(/\s+/);
    const nSig = parseInt(first[1], 10), fs = parseFloat(first[2]), nSamp = parseInt(first[3], 10);
    if (!(nSig > 0) || !(fs > 0) || !(nSamp > 0)) throw new Error("Could not read the .hea header line.");
    const sigs = [];
    for (let i = 1; i <= nSig; i++) {
      const p = (lines[i] || "").split(/\s+/);
      if (p.length < 9) throw new Error(`Malformed signal line ${i} in the .hea file.`);
      const fmt = parseInt(p[1], 10);                       // e.g. "16" or "16x2:0"
      const m = /^([\d.eE+-]+)(?:\((-?\d+)\))?(?:\/(.*))?$/.exec(p[2]);
      if (!m) throw new Error(`Cannot read the gain field "${p[2]}".`);
      const gain = parseFloat(m[1]) || 200;                 // WFDB default gain is 200 when 0
      const adcZero = parseInt(p[4], 10) || 0;
      const baseline = m[2] !== undefined ? parseInt(m[2], 10) : adcZero;
      sigs.push({ file: p[0], fmt, gain, baseline, name: p[p.length - 1] });
    }
    return { fs, nSamp, sigs };
  }

  function readWfdb(heaText, datBuffer) {
    const h = parseHeader(heaText);
    if (!h.sigs.every((s) => s.fmt === 16)) throw new Error("Only WFDB signal format 16 is supported (PTB-XL records use it).");
    if (new Set(h.sigs.map((s) => s.file)).size !== 1) throw new Error("All signals must be stored in one .dat file.");
    const n = h.sigs.length;
    if (datBuffer.byteLength < h.nSamp * n * 2) throw new Error("The .dat file is shorter than the header says; check that .hea and .dat belong together.");
    const dv = new DataView(datBuffer);
    const sig = h.sigs.map(() => new Float64Array(h.nSamp));
    for (let t = 0; t < h.nSamp; t++) {
      for (let s = 0; s < n; s++) {
        const d = dv.getInt16((t * n + s) * 2, true);        // little-endian, interleaved
        sig[s][t] = d === -32768 ? NaN : (d - h.sigs[s].baseline) / h.sigs[s].gain;
      }
    }
    return { fs: h.fs, names: h.sigs.map((s) => s.name), sig };
  }

  /* ---------------- filtering (scipy.signal.sosfiltfilt equivalent) ---------------- */
  function sosfilt(x, zi0) {
    const y = Float64Array.from(x);
    for (let s = 0; s < SOS.length; s++) {
      const [b0, b1, b2, , a1, a2] = SOS[s];
      let z0 = zi0[s][0], z1 = zi0[s][1];
      for (let n = 0; n < y.length; n++) {
        const xn = y[n], yn = b0 * xn + z0;
        z0 = b1 * xn - a1 * yn + z1;
        z1 = b2 * xn - a2 * yn;
        y[n] = yn;
      }
    }
    return y;
  }
  function bandpass(x) {
    const edge = 3 * (2 * SOS.length + 1), n = x.length;
    if (n <= edge) throw new Error("Signal too short to filter.");
    const ext = new Float64Array(n + 2 * edge);
    for (let i = 0; i < edge; i++) {
      ext[i] = 2 * x[0] - x[edge - i];                       // odd extension, left
      ext[edge + n + i] = 2 * x[n - 1] - x[n - 2 - i];       // odd extension, right
    }
    ext.set(x, edge);
    const scale = (c) => ZI.map((z) => [z[0] * c, z[1] * c]);
    let y = sosfilt(ext, scale(ext[0]));                       // forward pass, steady-state start
    const y0 = y[y.length - 1];
    y.reverse();
    y = sosfilt(y, scale(y0));                                 // backward pass
    y.reverse();
    return y.slice(edge, edge + n);
  }

  /* ---------------- statistics ---------------- */
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  const std0 = (a) => { const m = mean(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) * (v - m), 0) / a.length); };

  function statFeatures(x, p) {
    const n = x.length, m = mean(x);
    let m2 = 0, m3 = 0, m4 = 0, mn = Infinity, mx = -Infinity, sq = 0;
    for (const v of x) {
      const d = v - m;
      m2 += d * d; m3 += d * d * d; m4 += d * d * d * d; sq += v * v;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    m2 /= n; m3 /= n; m4 /= n;
    return {
      [`${p}_mean`]: m, [`${p}_std`]: Math.sqrt(m2), [`${p}_min`]: mn, [`${p}_max`]: mx, [`${p}_range`]: mx - mn,
      [`${p}_rms`]: Math.sqrt(sq / n), [`${p}_energy`]: sq / n,
      [`${p}_skew`]: m2 > 0 ? m3 / Math.pow(m2, 1.5) : NaN,
      [`${p}_kurtosis`]: m2 > 0 ? m4 / (m2 * m2) - 3 : NaN,
    };
  }

  /* ---------------- Welch PSD (scipy.signal.welch defaults, nperseg = 256) ---------------- */
  function welch(x, fs) {
    const N = 256, step = N / 2, nseg = Math.floor((x.length - N) / step) + 1;
    const w = new Float64Array(N);
    let wss = 0;
    for (let i = 0; i < N; i++) { w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N); wss += w[i] * w[i]; }   // periodic Hann
    const K = N / 2 + 1, psd = new Float64Array(K);
    const cosT = new Float64Array(K * N), sinT = new Float64Array(K * N);
    for (let k = 0; k < K; k++) for (let n = 0; n < N; n++) {
      const a = (2 * Math.PI * k * n) / N; cosT[k * N + n] = Math.cos(a); sinT[k * N + n] = Math.sin(a);
    }
    const seg = new Float64Array(N);
    for (let s = 0; s < nseg; s++) {
      let m = 0;
      for (let i = 0; i < N; i++) m += x[s * step + i];
      m /= N;
      for (let i = 0; i < N; i++) seg[i] = (x[s * step + i] - m) * w[i];          // constant detrend, window
      for (let k = 0; k < K; k++) {
        let re = 0, im = 0;
        for (let n = 0; n < N; n++) { re += seg[n] * cosT[k * N + n]; im -= seg[n] * sinT[k * N + n]; }
        psd[k] += re * re + im * im;
      }
    }
    const scale = 1 / (fs * wss * nseg);
    for (let k = 0; k < K; k++) psd[k] *= scale * (k === 0 || k === K - 1 ? 1 : 2);   // one-sided density
    return { freqs: Array.from({ length: K }, (_, k) => (k * fs) / N), psd };
  }

  function freqFeatures(x, p, fs) {
    const { freqs, psd } = welch(x, fs);
    const total = psd.reduce((s, v) => s + v, 0) + 1e-12, out = {};
    for (const [name, [lo, hi]] of Object.entries(BANDS)) {
      let s = 0;
      freqs.forEach((f, k) => { if (f >= lo && f < hi) s += psd[k]; });
      out[`${p}_bp_${name}`] = s / total;
    }
    let arg = 0, cen = 0, ent = 0;
    psd.forEach((v, k) => {
      if (v > psd[arg]) arg = k;
      const pr = v / total;
      cen += freqs[k] * pr;
      ent -= pr * Math.log2(pr + 1e-12);
    });
    out[`${p}_dom_freq`] = freqs[arg]; out[`${p}_spec_centroid`] = cen; out[`${p}_spec_entropy`] = ent;
    return out;
  }

  /* ---------------- find_peaks (scipy semantics: plateaus, height, distance) ---------------- */
  function findPeaks(x, distance, height) {
    let peaks = [];
    for (let i = 1; i < x.length - 1; i++) {
      if (x[i - 1] < x[i]) {
        let j = i + 1;
        while (j < x.length - 1 && x[j] === x[i]) j++;
        if (x[j] < x[i]) { peaks.push(Math.floor((i + j - 1) / 2)); i = j; }
      }
    }
    peaks = peaks.filter((p) => x[p] >= height);
    const keep = new Array(peaks.length).fill(true);
    const order = peaks.map((_, i) => i).sort((a, b) => (x[peaks[a]] - x[peaks[b]]) || (a - b));
    for (let i = peaks.length - 1; i >= 0; i--) {
      const j = order[i];
      if (!keep[j]) continue;
      for (let k = j - 1; k >= 0 && peaks[j] - peaks[k] < distance; k--) keep[k] = false;
      for (let k = j + 1; k < peaks.length && peaks[k] - peaks[j] < distance; k++) keep[k] = false;
    }
    return peaks.filter((_, i) => keep[i]);
  }

  const BEAT_KEYS = ["n_beats", "hr_mean", "rr_mean", "rr_std", "rmssd", "pnn50", "rr_cv", "r_amp_mean", "r_amp_std"];
  function beatFeatures(x, p, fs) {
    const out = {}, nan = () => { BEAT_KEYS.forEach((k) => { out[`${p}_${k}`] = NaN; }); return { out, peaks: [] }; };
    let mx = -Infinity, mn = Infinity;
    for (const v of x) { if (v > mx) mx = v; if (v < mn) mn = v; }
    const xp = Math.abs(mx) >= Math.abs(mn) ? x : x.map((v) => -v);
    const top = Math.abs(mx) >= Math.abs(mn) ? mx : -mn;          // max of xp
    if (top <= 0) return nan();
    const peaks = findPeaks(xp, Math.floor(0.25 * fs), 0.5 * top);
    if (peaks.length < 3) return nan();
    const rr = peaks.slice(1).map((v, i) => (v - peaks[i]) / fs);
    const drr = rr.slice(1).map((v, i) => v - rr[i]);
    const amps = peaks.map((i) => xp[i]), mrr = mean(rr);
    out[`${p}_n_beats`] = peaks.length; out[`${p}_hr_mean`] = 60 / mrr; out[`${p}_rr_mean`] = mrr;
    out[`${p}_rr_std`] = std0(rr);
    out[`${p}_rmssd`] = drr.length ? Math.sqrt(mean(drr.map((v) => v * v))) : NaN;
    out[`${p}_pnn50`] = drr.length ? mean(drr.map((v) => (Math.abs(v) > 0.05 ? 1 : 0))) : NaN;
    out[`${p}_rr_cv`] = std0(rr) / mrr;
    out[`${p}_r_amp_mean`] = mean(amps); out[`${p}_r_amp_std`] = std0(amps);
    return { out, peaks };
  }

  /* ---------------- main entry ---------------- */
  function extractFeatures(rec, leads, fullSetLeads) {
    if (Math.abs(rec.fs - FS) > 1e-9) {
      throw new Error(`This record is sampled at ${rec.fs} Hz. The model was trained on 100 Hz data: upload the “_lr” files (records100 folder).`);
    }
    const feats = {}, trace = {};
    for (const lead of leads) {
      const i = rec.names.indexOf(lead);
      if (i < 0) throw new Error(`Lead ${lead} not found in the record (${rec.names.join(", ")}).`);
      if (rec.sig[i].length < 256) throw new Error("The record is too short (at least 256 samples are needed).");
      const raw = rec.sig[i].map((v) => (Number.isNaN(v) ? 0 : v));
      const x = bandpass(raw);
      Object.assign(feats, statFeatures(x, lead), freqFeatures(x, lead, FS));
      if (fullSetLeads.includes(lead)) {
        const b = beatFeatures(x, lead, FS);
        Object.assign(feats, b.out);
        if (lead === "II") trace[lead] = { raw: Array.from(raw), filtered: Array.from(x), peaks: b.peaks };
      }
    }
    return { feats, trace };
  }

  const api = { readWfdb, parseHeader, bandpass, extractFeatures, findPeaks, welch };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ECGExtract = api;
})(typeof self !== "undefined" ? self : this);
