"""Step 3 - train / tune / evaluate / explain, then save the model bundle used by the Streamlit app.

Run after prepare_data.py:   python train.py
Outputs: models/*.joblib, data/demo_samples.csv, figures/*.png|csv
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")                       # save to files, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, PredefinedSplit, StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42
ROOT = Path(__file__).resolve().parent
FIG, MODELS = ROOT / "figures", ROOT / "models"
FIG.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
NOT_FEATURES = {"patient_id", "split", "abnormal", "superclasses", "filename_lr"}
ALL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# ------------------------------------------------------------------ load + checks
data = pd.read_csv(ROOT / "data" / "features.csv", index_col="ecg_id")
data["superclasses"] = data["superclasses"].str.split("|")
FEATURES = [c for c in data.columns if c not in NOT_FEATURES]
assert not ({"scp_codes", "report", "strat_fold", "patient_id", "abnormal"} & set(FEATURES)), "label/ID in features!"
LEADS = list(dict.fromkeys(c.split("_")[0] for c in FEATURES if c.split("_")[0] in ALL_LEADS))

tr, va, te = (data[data.split == s] for s in ("train", "val", "test"))
assert not (set(tr.patient_id) & set(va.patient_id) | set(tr.patient_id) & set(te.patient_id)
            | set(va.patient_id) & set(te.patient_id)), "PATIENT LEAKAGE between splits"
print(f"Leakage check passed. train/val/test = {len(tr)}/{len(va)}/{len(te)} records, {len(FEATURES)} features")
X_tr, y_tr, X_va, y_va, X_te, y_te = tr[FEATURES], tr["abnormal"], va[FEATURES], va["abnormal"], te[FEATURES], te["abnormal"]
X_trva, y_trva = pd.concat([X_tr, X_va]), pd.concat([y_tr, y_va])

# ------------------------------------------------------------------ models + tuning on the validation fold
pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
models = {
    "Logistic Regression": (
        Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                  ("clf", LogisticRegression(class_weight="balanced", max_iter=3000))]),
        {"clf__C": [0.01, 0.1, 1, 10]}),
    "Random Forest": (
        Pipeline([("imp", SimpleImputer(strategy="median")),
                  ("clf", RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=SEED))]),
        {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 12], "clf__min_samples_leaf": [1, 5]}),
    "XGBoost": (
        XGBClassifier(scale_pos_weight=pos_weight, eval_metric="logloss", tree_method="hist",
                      n_jobs=-1, random_state=SEED),
        {"n_estimators": [200, 400], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]}),
}
ps = PredefinedSplit(np.r_[np.full(len(X_tr), -1), np.zeros(len(X_va), dtype=int)])   # -1 = train, 0 = validation
fitted, val_auc = {}, {}
for name, (est, grid) in models.items():
    gs = GridSearchCV(est, grid, scoring="roc_auc", cv=ps, refit=True).fit(X_trva, y_trva)
    fitted[name], val_auc[name] = gs.best_estimator_, gs.best_score_
    print(f"{name:20s} validation AUC = {gs.best_score_:.4f} | {gs.best_params_}")

# patient-grouped CV on the training part (robustness evidence for the report)
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_rows = []
for name, m in fitted.items():
    s = cross_val_score(m, X_tr, y_tr, groups=tr["patient_id"].values, cv=cv, scoring="roc_auc")
    cv_rows.append({"Model": name, "CV ROC-AUC mean": s.mean(), "CV ROC-AUC std": s.std()})
cv_table = pd.DataFrame(cv_rows).set_index("Model").round(4)
cv_table.to_csv(FIG / "cv_results.csv")
print("\nPatient-grouped 5-fold CV (train part):\n", cv_table.to_string())

# ------------------------------------------------------------------ evaluation on the untouched test fold
def auc_ci(y, p, n=1000):
    rng = np.random.default_rng(SEED); y = np.asarray(y); aucs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            aucs.append(roc_auc_score(y[i], p[i]))
    return np.percentile(aucs, [2.5, 97.5])

rows, probs = [], {}
for name, m in fitted.items():
    p = m.predict_proba(X_te)[:, 1]; probs[name] = p
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    lo, hi = auc_ci(y_te, p)
    rows.append({"Model": name, "Accuracy": accuracy_score(y_te, pred), "Precision": precision_score(y_te, pred),
                 "Recall (sensitivity)": recall_score(y_te, pred), "Specificity": tn / (tn + fp),
                 "F1": f1_score(y_te, pred), "ROC-AUC": roc_auc_score(y_te, p),
                 "AUC 95% CI": f"{lo:.3f}-{hi:.3f}", "PR-AUC": average_precision_score(y_te, p)})
results = pd.DataFrame(rows).set_index("Model").round(4)
results.to_csv(FIG / "model_comparison.csv")
print("\nTest-fold results:\n", results.to_string())

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, (name, p) in zip(ax, probs.items()):
    ConfusionMatrixDisplay.from_predictions(y_te, (p >= 0.5).astype(int), display_labels=["Normal", "Abnormal"],
                                            ax=a, colorbar=False, cmap="Blues")
    a.set_title(name)
plt.tight_layout(); plt.savefig(FIG / "confusion_matrices.png", dpi=200); plt.close()

plt.figure(figsize=(6, 5))
for name, p in probs.items():
    fpr, tpr, _ = roc_curve(y_te, p)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_te, p):.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False positive rate (1 - specificity)"); plt.ylabel("True positive rate (recall)")
plt.title("ROC curves - test fold"); plt.legend(loc="lower right"); plt.tight_layout()
plt.savefig(FIG / "roc_curves.png", dpi=200); plt.close()

# ------------------------------------------------------------------ error analysis
best_name = max(val_auc, key=val_auc.get)
err = te.assign(prob=probs[best_name]); err["pred"] = (err["prob"] >= 0.5).astype(int)
ab = err[err.abnormal == 1]
classes = ["MI", "STTC", "CD", "HYP"]
per_class = pd.DataFrame({
    "n_test_records": {c: int(ab.superclasses.apply(lambda s, c=c: c in s).sum()) for c in classes},
    "recall": {c: ab[ab.superclasses.apply(lambda s, c=c: c in s)].pred.mean() for c in classes},
}).round(3)
per_class.to_csv(FIG / "recall_per_superclass.csv")
print(f"\nBest model by validation AUC: {best_name}")
print("Recall per diagnostic superclass:\n", per_class.to_string())
print("False positives:", int(((err.abnormal == 0) & (err.pred == 1)).sum()),
      "| False negatives:", int(((err.abnormal == 1) & (err.pred == 0)).sum()))

# ------------------------------------------------------------------ explainability
xgb = fitted["XGBoost"]
sv = shap.TreeExplainer(xgb)(X_te)                       # contributions in log-odds
shap.plots.beeswarm(sv, max_display=15, show=False)
plt.savefig(FIG / "shap_beeswarm.png", dpi=200, bbox_inches="tight"); plt.close()
shap.plots.bar(sv, max_display=15, show=False)
plt.savefig(FIG / "shap_bar.png", dpi=200, bbox_inches="tight"); plt.close()
i = int(np.argmax(probs["XGBoost"] * (y_te.values == 1)))      # most confident true positive
shap.plots.waterfall(sv[i], max_display=10, show=False)
plt.savefig(FIG / "shap_waterfall_example.png", dpi=200, bbox_inches="tight"); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(16, 5))
for a, (name, m) in zip(ax, fitted.items()):
    r = permutation_importance(m, X_te, y_te, scoring="roc_auc", n_repeats=10, random_state=SEED)
    pd.Series(r.importances_mean, index=FEATURES).sort_values().tail(10).plot.barh(ax=a, title=name)
    a.set_xlabel("Drop in ROC-AUC")
plt.tight_layout(); plt.savefig(FIG / "permutation_importance.png", dpi=200); plt.close()

# ------------------------------------------------------------------ save for the Streamlit app
joblib.dump({"model": xgb, "features": FEATURES, "leads": LEADS, "dataset": "PTB-XL v1.0.3"},
            MODELS / "xgb_bundle.joblib")
for name, m in fitted.items():
    joblib.dump(m, MODELS / f"{name.lower().replace(' ', '_')}.joblib")

demo = pd.concat([te[te.abnormal == 0].head(3), te[te.abnormal == 1].head(3)])     # held-out records only
demo_out = demo[FEATURES].copy()
demo_out["true_label"] = demo["abnormal"].map({0: "Normal", 1: "Abnormal"})
demo_out.to_csv(ROOT / "data" / "demo_samples.csv")

bundle = joblib.load(MODELS / "xgb_bundle.joblib")                                    # self-test
pp = bundle["model"].predict_proba(demo[bundle["features"]])[:, 1]
assert ((pp >= 0) & (pp <= 1)).all()
print("\nSaved models/, data/demo_samples.csv and figures/. Reload test OK:", pp.round(2))
print("Next:  python export_web.py")
