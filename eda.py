"""Step 2 - exploratory data analysis. Saves figures to figures/ (run after prepare_data.py)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # save to files, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wfdb

from ecg_features import bandpass

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
RAW = ROOT / "data" / "raw"
NOT_FEATURES = {"patient_id", "split", "abnormal", "superclasses", "filename_lr"}

data = pd.read_csv(ROOT / "data" / "features.csv", index_col="ecg_id")
data["superclasses"] = data["superclasses"].str.split("|")
FEATURES = [c for c in data.columns if c not in NOT_FEATURES]
train = data[data.split == "train"]
label = data["abnormal"].map({0: "Normal", 1: "Abnormal"})

# 1. class balance and diagnostic superclasses
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.countplot(x=label, hue=data["split"], ax=ax[0])
ax[0].set_title("Class distribution per split"); ax[0].set_xlabel("")
data["superclasses"].explode().value_counts().plot.bar(ax=ax[1], title="Diagnostic superclasses in the subset")
plt.tight_layout(); plt.savefig(FIG / "eda_classes.png", dpi=200); plt.close()

# 2. before / after preprocessing (raw lead II vs 0.5-40 Hz band-pass)
ids = [data[data.abnormal == 0].index[0], data[data.abnormal == 1].index[0]]
fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
for a, i in zip(ax, ids):
    sig, fld = wfdb.rdsamp(str(RAW / data.loc[i, "filename_lr"]))
    raw = np.nan_to_num(sig[:, fld["sig_name"].index("II")])
    t = np.arange(len(raw)) / fld["fs"]
    a.plot(t, raw, alpha=0.5, label="raw")
    a.plot(t, bandpass(raw), label="band-pass 0.5-40 Hz")
    a.set_title(f"ECG {i} - {label[i]} (lead II)"); a.set_ylabel("mV"); a.legend(loc="upper right")
ax[-1].set_xlabel("seconds")
plt.tight_layout(); plt.savefig(FIG / "eda_before_after_filter.png", dpi=200); plt.close()

# 3. a few feature distributions by class
show = [f for f in ["II_hr_mean", "II_rr_cv", "II_range", "II_rms", "II_bp_qrs", "II_spec_entropy"] if f in FEATURES]
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for a, f in zip(axes.ravel(), show):
    sns.boxplot(x=label[train.index], y=train[f], ax=a, showfliers=False)
    a.set_xlabel(""); a.set_title(f)
plt.tight_layout(); plt.savefig(FIG / "eda_feature_boxplots.png", dpi=200); plt.close()

# 4. correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(train[FEATURES].corr(), cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": 0.7})
plt.title("Feature correlation (train)"); plt.tight_layout()
plt.savefig(FIG / "eda_correlation.png", dpi=200); plt.close()

# 5. text summaries for the report
print("Rows per split:\n", data["split"].value_counts().to_string())
print("\nMissing values (fraction, top 5):\n", train[FEATURES].isna().mean().sort_values(ascending=False).head(5).round(3).to_string())
train[FEATURES].describe().T.round(3).to_csv(FIG / "feature_summary.csv")
print(f"\nSaved figures to {FIG}")
