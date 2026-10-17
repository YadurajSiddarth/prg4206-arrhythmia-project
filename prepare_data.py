"""Step 1 - build the feature table.

metadata -> normal/abnormal label -> official-fold split (+ leakage checks)
-> download ONLY the sampled 100 Hz records -> feature extraction -> data/features.csv

Usage:
    python prepare_data.py                                  # 10,000 records, lead II
    python prepare_data.py --n-train 3000 --n-val 500 --n-test 500
    python prepare_data.py --leads II V1 V5                 # features from several leads
    python prepare_data.py --force                          # rebuild even if features.csv exists
"""
import argparse
import ast
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
import wfdb
from joblib import Parallel, delayed
from tqdm import tqdm

from ecg_features import extract_features

BASE = "https://physionet.org/files/ptb-xl/1.0.3"
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"            # metadata + downloaded records live here
OUT = ROOT / "data" / "features.csv"
SEED = 42


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=7000, help="records from folds 1-8")
    p.add_argument("--n-val", type=int, default=1500, help="records from fold 9")
    p.add_argument("--n-test", type=int, default=1500, help="records from fold 10")
    p.add_argument("--leads", nargs="+", default=["II"])
    p.add_argument("--workers", type=int, default=16, help="parallel download threads")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def get_file(session, rel):
    """Download one file from PhysioNet into data/raw (skipped if it already exists)."""
    dst = RAW / rel
    if dst.exists() and dst.stat().st_size > 0:
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            r = session.get(f"{BASE}/{rel}", timeout=30)
            r.raise_for_status()
            dst.write_bytes(r.content)
            return None
        except Exception as exc:                       # network hiccup -> retry
            if attempt == 3:
                return f"{rel}: {exc}"
            time.sleep(1 + attempt)


def build_labels():
    """Read metadata and create the binary target (0 = NORM only, 1 = any other diagnostic superclass)."""
    session = requests.Session()
    for fn in ("ptbxl_database.csv", "scp_statements.csv"):   # drop your own copies in data/raw/ to skip this
        err = get_file(session, fn)
        if err:
            raise SystemExit(f"Could not download {err}")

    Y = pd.read_csv(RAW / "ptbxl_database.csv", index_col="ecg_id")
    Y["scp_codes"] = Y["scp_codes"].apply(ast.literal_eval)
    scp = pd.read_csv(RAW / "scp_statements.csv", index_col=0)
    scp = scp[scp["diagnostic"] == 1]                          # diagnostic statements only
    print(f"All records: {len(Y)} | patients: {Y.patient_id.nunique()}")

    def to_superclasses(codes):
        return sorted({scp.loc[c, "diagnostic_class"] for c in codes if c in scp.index})

    Y["superclasses"] = Y["scp_codes"].apply(to_superclasses)
    Y = Y[Y["superclasses"].str.len() > 0].copy()              # drop records without a diagnostic label
    Y["abnormal"] = Y["superclasses"].apply(lambda s: int(s != ["NORM"]))
    print(f"Labelled records: {len(Y)} | abnormal share: {Y['abnormal'].mean():.1%}")
    return Y


def split_by_official_folds(Y, n_train, n_val, n_test):
    def take(df, n):
        return df if n >= len(df) else df.sample(n=n, random_state=SEED)

    meta = pd.concat([
        take(Y[Y.strat_fold <= 8], n_train).assign(split="train"),
        take(Y[Y.strat_fold == 9], n_val).assign(split="val"),
        take(Y[Y.strat_fold == 10], n_test).assign(split="test"),
    ])
    # ---- leakage checks: a patient must never appear in two splits ----
    pid = {s: set(meta.loc[meta.split == s, "patient_id"]) for s in ("train", "val", "test")}
    assert not (pid["train"] & pid["val"]), "patient overlap: train/val"
    assert not (pid["train"] & pid["test"]), "patient overlap: train/test"
    assert not (pid["val"] & pid["test"]), "patient overlap: val/test"
    assert meta.index.is_unique, "duplicate ecg_id"
    print("Leakage check passed: no patient appears in more than one split.")
    print(meta.groupby("split")["abnormal"].agg(records="count", abnormal_rate="mean").round(3))
    return meta


def download_records(meta, workers):
    session = requests.Session()
    rels = [f"{fn}{ext}" for fn in meta["filename_lr"] for ext in (".hea", ".dat")]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        errors = [e for e in tqdm(ex.map(lambda r: get_file(session, r), rels), total=len(rels),
                                  desc="Downloading records") if e]
    print(f"Download finished in {time.time() - t0:.0f}s, failed files: {len(errors)}")
    return errors


def process(ecg_id, filename_lr, leads):
    try:
        sig, fields = wfdb.rdsamp(str(RAW / filename_lr))
        feats = extract_features(sig, fields["sig_name"], fields["fs"], leads=leads, full_set_leads=("II",))
        feats["ecg_id"] = ecg_id
        return feats
    except Exception as exc:                                   # unreadable / corrupt record
        return {"ecg_id": ecg_id, "_error": str(exc)}


def main():
    args = parse_args()
    if OUT.exists() and not args.force:
        print(f"{OUT} already exists - nothing to do (use --force to rebuild).")
        return
    RAW.mkdir(parents=True, exist_ok=True)

    Y = build_labels()
    meta = split_by_official_folds(Y, args.n_train, args.n_val, args.n_test)
    download_records(meta, args.workers)

    rows = Parallel(n_jobs=-1)(delayed(process)(i, fn, args.leads)
                               for i, fn in tqdm(zip(meta.index, meta.filename_lr), total=len(meta),
                                                 desc="Extracting features"))
    feats = pd.DataFrame(rows).set_index("ecg_id")
    if "_error" in feats.columns:
        print(f"Unreadable / missing records skipped: {int(feats['_error'].notna().sum())}")
        feats = feats[feats["_error"].isna()].drop(columns="_error")

    info = meta[["patient_id", "split", "abnormal", "filename_lr"]].copy()
    info["superclasses"] = meta["superclasses"].apply("|".join)
    data = feats.join(info, how="inner")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT)
    print(f"Saved {OUT} with shape {data.shape}")


if __name__ == "__main__":
    main()
