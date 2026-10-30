# Normal vs Abnormal ECG classification (PTB-XL, explainable ML)

Binary classifier (normal vs abnormal ECG) on lead-II features from PTB-XL v1.0.3: Logistic Regression, Random Forest and
XGBoost, SHAP explanations, and a static web prototype that runs the model **in the browser** (no server needed).

> **Not a clinical diagnostic tool.** Student research prototype only.

**Live demo:** `https://yadurajsiddarth.github.io/prg4206-arrhythmia-project/` (GitHub Pages, served from the `docs/` folder)

## Pipeline (macOS / Linux)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python prepare_data.py   # metadata -> labels -> official folds -> download only the sampled records -> data/features.csv
python eda.py            # figures/eda_*.png
python train.py          # trains 3 models, evaluation + SHAP figures, models/*.joblib
python export_web.py     # checks and exports the XGBoost trees to docs/model.js
```
Windows: activate with `.venv\Scripts\activate`.

## Run the prototype locally
Open `docs/index.html` with the VS Code **Live Server** extension (right-click > Open with Live Server).
`model.js` is loaded as a normal script, so the page also opens by double-clicking `docs/index.html`.

## Method summary
* Label: diagnostic superclass NORM only = normal; any other diagnostic superclass = abnormal.
* Split: official PTB-XL folds (1-8 train, 9 validation, 10 test); a patient never appears in two splits (asserted in code).
* Features: statistics, energy, Welch band powers and R-peak / HRV features from band-passed (0.5-40 Hz) lead II at 100 Hz.
* Models are tuned on the validation fold; metrics are reported on the test fold; SHAP and permutation importance explain them.
* Web prototype: `export_web.py` verifies that the exported trees reproduce XGBoost's predictions before writing `docs/model.js`.
  The page computes exact TreeSHAP values in JavaScript (`docs/engine.js`) and re-checks itself against XGBoost's own SHAP
  output on load (see the green badge at the bottom of the page). `node tests/test_engine.js` checks the JavaScript TreeSHAP
  against brute-force Shapley values.
* The page also accepts a raw PTB-XL record (`.hea` + `.dat`, 100 Hz `_lr` files). `docs/ecg-extract.js` is a JavaScript port of
  `ecg_features.py` (Butterworth band-pass, Welch spectrum, R-peak detection); `python tests/test_extract.py` verifies that it
  reproduces the Python features on synthetic WFDB records.

## Data and licence
PTB-XL: Wagner et al., Scientific Data 2020; PhysioNet, CC BY 4.0. Raw records are not stored in this repository.
