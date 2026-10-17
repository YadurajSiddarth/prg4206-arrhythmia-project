// Compares the JavaScript feature extractor with the Python reference values written by test_extract.py
const fs = require("fs"), path = require("path");
const X = require("../docs/ecg-extract.js");
const dir = process.argv[2];
const expected = JSON.parse(fs.readFileSync(path.join(dir, "expected.json"), "utf8"));
let worst = 0, bad = 0, total = 0;
for (const [name, exp] of Object.entries(expected)) {
  const hea = fs.readFileSync(path.join(dir, name + ".hea"), "utf8");
  const dat = fs.readFileSync(path.join(dir, name + ".dat"));
  const rec = X.readWfdb(hea, dat.buffer.slice(dat.byteOffset, dat.byteOffset + dat.byteLength));
  const { feats } = X.extractFeatures(rec, ["II", "V1"], ["II"]);
  for (const [k, ev] of Object.entries(exp)) {
    total++;
    const gv = feats[k];
    if (ev === null) { if (!Number.isNaN(gv)) { bad++; console.log(`  ${name}.${k}: expected NaN, got ${gv}`); } continue; }
    if (gv === undefined || Number.isNaN(gv)) { bad++; console.log(`  ${name}.${k}: expected ${ev}, got ${gv}`); continue; }
    const err = Math.abs(gv - ev) / Math.max(1e-9, Math.abs(ev));
    worst = Math.max(worst, err);
    if (err > 1e-6) { bad++; console.log(`  ${name}.${k}: expected ${ev}, got ${gv} (rel err ${err.toExponential(1)})`); }
  }
}
console.log(`compared ${total} features over ${Object.keys(expected).length} records, worst relative error ${worst.toExponential(2)}, mismatches ${bad}`);
process.exit(bad ? 1 : 0);
