"""Streamlit prototype: normal vs abnormal ECG (PTB-XL, extracted features, XGBoost + SHAP).
Run:  streamlit run app.py
"""
import os
import tempfile

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from ecg_features import extract_features

HERE = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="ECG Normal vs Abnormal", layout="wide")


@st.cache_resource
def load_bundle():
    bundle = joblib.load(os.path.join(HERE, "models", "xgb_bundle.joblib"))
    return bundle, shap.TreeExplainer(bundle["model"])


def features_from_wfdb(files, leads):
    """Build one feature row from an uploaded WFDB record (.hea + .dat)."""
    import wfdb
    with tempfile.TemporaryDirectory() as tmp:
        for up in files:
            with open(os.path.join(tmp, up.name), "wb") as fh:
                fh.write(up.getbuffer())
        stems = [u.name[:-4] for u in files if u.name.endswith(".hea")]
        if not stems:
            raise ValueError("Please upload the .hea file together with the .dat file.")
        sig, fld = wfdb.rdsamp(os.path.join(tmp, stems[0]))
        return extract_features(sig, fld["sig_name"], fld["fs"], leads=leads, full_set_leads=("II",))


bundle, explainer = load_bundle()
FEATURES = bundle["features"]

st.title("ECG Normal vs Abnormal - research prototype")
st.warning(
    "**This is NOT a clinical diagnostic tool.** It is a student research prototype trained on the "
    "PTB-XL dataset and must never be used to make medical decisions."
)

mode = st.radio("Input type", ["Demo sample", "Upload feature CSV", "Upload WFDB record (.hea + .dat)"],
                horizontal=True)
X, truth = None, None
try:
    if mode == "Demo sample":
        demo = pd.read_csv(os.path.join(HERE, "data", "demo_samples.csv"), index_col="ecg_id")
        choice = st.selectbox("Held-out test record (ECG id)", demo.index)
        X = demo.loc[[choice], FEATURES]
        truth = demo.loc[choice, "true_label"] if "true_label" in demo.columns else None
    elif mode == "Upload feature CSV":
        up = st.file_uploader("CSV with the model's feature columns (first row is used)", type="csv")
        if up is not None:
            df = pd.read_csv(up)
            missing = [c for c in FEATURES if c not in df.columns]
            if missing:
                st.error(f"Missing {len(missing)} feature columns, e.g. {missing[:3]}")
            else:
                X = df[FEATURES].iloc[[0]]
    else:
        ups = st.file_uploader("Upload BOTH files of one record", accept_multiple_files=True)
        if ups and len(ups) >= 2:
            X = pd.DataFrame([features_from_wfdb(ups, bundle["leads"])])[FEATURES]
except Exception as exc:  # show a friendly message instead of a stack trace
    st.error(f"Could not read the input: {exc}")

if X is not None:
    prob = float(bundle["model"].predict_proba(X)[0, 1])
    label = "ABNORMAL" if prob >= 0.5 else "NORMAL"
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted class", label)
    c2.metric("Probability of abnormal", f"{prob:.1%}")
    c3.metric("Confidence in predicted class", f"{max(prob, 1 - prob):.1%}")
    st.progress(min(max(prob, 0.0), 1.0))
    if truth is not None:
        st.caption(f"Ground-truth label (demo only): {truth}")

    sv = explainer(X)
    st.subheader("Which features influenced this prediction?")
    st.caption("SHAP values are in log-odds: positive pushes towards ABNORMAL, negative towards NORMAL.")
    plt.figure()
    shap.plots.waterfall(sv[0], max_display=10, show=False)
    st.pyplot(plt.gcf(), bbox_inches="tight")
    top = pd.DataFrame({"feature": FEATURES, "value": X.iloc[0].values, "SHAP (log-odds)": sv.values[0]})
    top = top.reindex(top["SHAP (log-odds)"].abs().sort_values(ascending=False).index).head(8)
    st.dataframe(top.reset_index(drop=True))
