"""Step 4 - export the trained XGBoost model to docs/model.js for the static web prototype.

Run after train.py:   python export_web.py

The browser page (docs/index.html) runs the trees itself, so no Python server is needed and the
page can be served by VS Code Live Server or GitHub Pages. This script CHECKS the export before
writing it: the exported trees must reproduce XGBoost's own margins and probabilities.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SEED = 42


def sig(x, digits=9):
    return float(f"{x:.{digits}g}")


def compact_tree(tree_json, features):
    """XGBoost JSON dump of one tree -> parallel arrays (node 0 = root)."""
    nodes = {}

    def walk(n):
        nodes[n["nodeid"]] = n
        for ch in n.get("children", []):
            walk(ch)

    walk(tree_json)
    ids = sorted(nodes)
    pos = {nid: i for i, nid in enumerate(ids)}
    assert pos[tree_json["nodeid"]] == 0, "root must be node 0"

    def feat_index(name):
        if name in features:
            return features.index(name)
        m = re.fullmatch(r"f(\d+)", name)          # model was fitted on a numpy array
        if m:
            return int(m.group(1))
        raise ValueError(f"unknown feature name in tree: {name}")

    T = {k: [] for k in "ftlrmcv"}
    for nid in ids:
        n = nodes[nid]
        if "leaf" in n:
            T["f"].append(-1); T["t"].append(0); T["l"].append(-1); T["r"].append(-1); T["m"].append(-1)
            T["c"].append(sig(n["cover"], 7)); T["v"].append(sig(n["leaf"], 8))
        else:
            T["f"].append(feat_index(n["split"])); T["t"].append(sig(n["split_condition"], 9))
            T["l"].append(pos[n["yes"]]); T["r"].append(pos[n["no"]]); T["m"].append(pos[n["missing"]])
            T["c"].append(sig(n["cover"], 7)); T["v"].append(0)
    return T


def py_leaf(tree, x):
    n = 0
    while tree["f"][n] >= 0:
        xv = x[tree["f"][n]]
        if np.isnan(xv):
            n = tree["m"][n]
        else:
            n = tree["l"][n] if np.float32(xv) < np.float32(tree["t"][n]) else tree["r"][n]
    return tree["v"][n]


def py_tree_sum(trees, x):
    return sum(py_leaf(t, x) for t in trees)


def nan_to_none(v):
    return None if (isinstance(v, float) and np.isnan(v)) else v


def main():
    import joblib
    import xgboost as xgb

    bundle = joblib.load(ROOT / "models" / "xgb_bundle.joblib")
    model, FEATURES = bundle["model"], list(bundle["features"])
    booster = model.get_booster()

    trees = [compact_tree(json.loads(d), FEATURES) for d in booster.get_dump(dump_format="json", with_stats=True)]
    print(f"Exported {len(trees)} trees, {sum(len(t['f']) for t in trees)} nodes")

    data = pd.read_csv(ROOT / "data" / "features.csv", index_col="ecg_id")
    test = data[data.split == "test"]
    train = data[data.split == "train"]
    X = test[FEATURES]

    # ---- check 1: exported trees reproduce XGBoost margins (offset = base score in log-odds) ----
    check = X.sample(n=min(800, len(X)), random_state=SEED)
    ref_margin = booster.predict(xgb.DMatrix(check), output_margin=True)
    sums = np.array([py_tree_sum(trees, row) for row in check.to_numpy(dtype=float)])
    diff = ref_margin - sums
    offset = float(np.median(diff))
    assert np.abs(diff - offset).max() < 1e-3, f"tree export does not match XGBoost (max dev {np.abs(diff - offset).max():.2e})"
    prob = 1 / (1 + np.exp(-(sums + offset)))
    p_err = np.abs(prob - model.predict_proba(check)[:, 1]).max()
    assert p_err < 1e-3, f"probability mismatch {p_err:.2e}"
    print(f"Check 1 passed: margin deviation {np.abs(diff - offset).max():.1e}, probability deviation {p_err:.1e}")

    # ---- self-test vectors: xgboost's own SHAP values for a few records (the page re-checks these on load) ----
    st = X.head(5)
    contribs = booster.predict(xgb.DMatrix(st), pred_contribs=True)       # (n, features + bias)
    selftest = [{"x": [nan_to_none(float(v)) for v in row],
                 "margin": float(c.sum()), "phi": [float(v) for v in c[:-1]], "bias": float(c[-1])}
                for row, c in zip(st.to_numpy(dtype=float), contribs)]

    # ---- demo records (held-out test split) + typical values for the manual form ----
    demo = pd.concat([test[test.abnormal == 0].sample(5, random_state=SEED),
                      test[test.abnormal == 1].sample(5, random_state=SEED)])
    demo_list = [{"id": int(i), "label": "Abnormal" if r["abnormal"] else "Normal",
                  "superclasses": r["superclasses"].replace("|", ", "),
                  "x": [nan_to_none(sig(float(r[f]), 6)) if not pd.isna(r[f]) else None for f in FEATURES]}
                 for i, r in demo.iterrows()]
    stats = {f: {"median": sig(float(train[f].median()), 6), "p1": sig(float(train[f].quantile(0.01)), 6),
                 "p99": sig(float(train[f].quantile(0.99)), 6)} for f in FEATURES}

    metrics = []
    mc = ROOT / "figures" / "model_comparison.csv"
    if mc.exists():
        metrics = json.loads(pd.read_csv(mc).to_json(orient="records"))

    out = {
        "features": FEATURES, "offset": offset, "trees": trees, "stats": stats, "demo": demo_list,
        "selftest": selftest, "metrics": metrics,
        "meta": {"dataset": "PTB-XL v1.0.3", "leads": bundle["leads"], "n_trees": len(trees),
                 "n_train": int((data.split == "train").sum()), "n_val": int((data.split == "val").sum()),
                 "n_test": int((data.split == "test").sum())},
    }
    dest = ROOT / "docs" / "model.js"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text("window.ECG_MODEL = " + json.dumps(out, separators=(",", ":"), allow_nan=False) + ";\n")
    print(f"Wrote {dest} ({dest.stat().st_size / 1e6:.2f} MB)")
    print("Next: open docs/index.html with VS Code Live Server ('Go Live').")


if __name__ == "__main__":
    main()
