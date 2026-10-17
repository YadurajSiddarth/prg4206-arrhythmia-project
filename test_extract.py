"""Checks that docs/ecg-extract.js (browser) reproduces ecg_features.py (Python) on synthetic WFDB records.
Run from the project root:  python tests/test_extract.py     (needs Node.js)"""
import json, subprocess, sys, tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ecg_features import extract_features          # noqa: E402

NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS = 100


def synth(seed, hr=70, jitter=0.0, polarity=1, noise=0.02, flat=False):
    rng = np.random.default_rng(seed)
    t = np.arange(FS * 10) / FS
    cols = []
    for lead in range(12):
        x = 0.1 * np.sin(2 * np.pi * 0.25 * t + lead) + noise * rng.standard_normal(len(t))
        beat = np.cumsum(np.full(30, 60 / hr) + jitter * rng.standard_normal(30)) - 0.3
        for b in beat:
            for off, amp, w in [(-0.2, 0.12, 0.04), (-0.03, -0.12, 0.012), (0.0, 1.0, 0.015), (0.03, -0.2, 0.012), (0.27, 0.3, 0.06)]:
                x += polarity * amp * (1 + 0.1 * lead) * np.exp(-0.5 * ((t - b - off) / w) ** 2)
        cols.append(np.full(len(t), 0.3) if flat else x)
    return np.stack(cols, axis=1)


def write_wfdb(folder, name, sig, gain_spec, comment=True):
    gain = float(gain_spec.split("(")[0].split("/")[0])
    baseline = int(gain_spec.split("(")[1].split(")")[0]) if "(" in gain_spec else 0
    digital = np.clip(np.round(sig * gain + baseline), -32767, 32767).astype("<i2")
    digital[5, [1, 3, 6]] = -32768                          # invalid samples in II, aVR, V1 (NaN -> 0 in both pipelines)
    (folder / f"{name}.dat").write_bytes(digital.tobytes())             # interleaved little-endian int16
    lines = [f"{name} 12 100 {sig.shape[0]}"]
    lines += [f"{name}.dat 16 {gain_spec} 16 0 -56 -22 0 {n}" for n in NAMES]
    if comment:
        lines += ["# Age: 56", "# Sex: 1"]
    (folder / f"{name}.hea").write_text("\n".join(lines) + "\n")
    phys = (digital.astype(float) - baseline) / gain
    phys[digital == -32768] = np.nan
    return phys


cases = {
    "normal": dict(sig=synth(1), gain="1000.0(0)/mV"),
    "fast_irregular": dict(sig=synth(2, hr=110, jitter=0.12, noise=0.04), gain="1000/mV"),
    "inverted": dict(sig=synth(3, polarity=-1), gain="1000(5)/mV"),         # negative R waves + explicit baseline
    "noisy_slow": dict(sig=synth(4, hr=48, noise=0.12), gain="500.0(0)/mV"),
    "flat_lead": dict(sig=synth(5, flat=True), gain="1000/mV"),              # constant signal -> NaN features
}
import os
tmp = Path(os.environ.get('ECG_TEST_DIR') or tempfile.mkdtemp())
tmp.mkdir(exist_ok=True)
expected = {}
for name, c in cases.items():
    phys = write_wfdb(tmp, name, c["sig"], c["gain"])
    f = extract_features(phys, NAMES, FS, leads=("II", "V1"), full_set_leads=("II",))
    expected[name] = {k: (None if (isinstance(v, float) and np.isnan(v)) else float(v)) for k, v in f.items()}
(tmp / "expected.json").write_text(json.dumps(expected))
res = subprocess.run(["node", str(ROOT / "tests" / "test_extract.js"), str(tmp)], capture_output=True, text=True)
print(res.stdout.strip(), res.stderr.strip())
sys.exit(res.returncode)
