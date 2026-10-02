import os
import ast
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import wfdb
from tqdm import tqdm

# 1. Configuration Setup
DATA_DIR = './data/ptb-xl/'
MAX_SAMPLES = 2500  # Extracting 2,500 records keeps local processing time low while providing enough data for ML

print("Loading metadata...")
df_meta = pd.read_csv(os.path.join(DATA_DIR, 'ptbxl_database.csv'), index_col='ecg_id')
df_scp = pd.read_csv(os.path.join(DATA_DIR, 'scp_statements.csv'), index_col=0)

# Parse SCP codes from strings to dictionaries
df_meta.scp_codes = df_meta.scp_codes.apply(lambda x: ast.literal_eval(x))
diagnostic_scp = df_scp[df_scp.diagnostic == 1]

# 2. Assign Primary Target Labels
def get_primary_label(y_dic):
    """Extracts the most prominent diagnostic superclass for classification."""
    for key in y_dic.keys():
        if key in diagnostic_scp.index:
            cls = diagnostic_scp.loc[key].diagnostic_class
            if pd.notna(cls):
                return cls
    return 'NORM'

df_meta['label'] = df_meta.scp_codes.apply(get_primary_label)
df_subset = df_meta.head(MAX_SAMPLES).copy()

# 3. Define Feature Extraction Logic
def extract_ecg_features(signal):
    """Calculates time-domain statistics for all 12 electrical leads."""
    features = []
    for lead in range(12):
        lead_sig = signal[:, lead]
        # Extract 8 specific statistical features per lead
        features.extend([
            np.mean(lead_sig),
            np.std(lead_sig),
            np.max(lead_sig),
            np.min(lead_sig),
            np.median(lead_sig),
            skew(lead_sig),
            kurtosis(lead_sig),
            np.sum(np.square(lead_sig))  # Signal Energy
        ])
    return features

# Generate the 96 column names (12 leads * 8 metrics)
feature_names = []
leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
metrics = ['mean', 'std', 'max', 'min', 'median', 'skew', 'kurtosis', 'energy']
for lead in leads:
    for metric in metrics:
        feature_names.append(f"{lead}_{metric}")

X_list = []
y_list = []

# 4. Process the Waveforms
print(f"Extracting features from {MAX_SAMPLES} records. This will take a few minutes...")
for ecg_id, row in tqdm(df_subset.iterrows(), total=len(df_subset)):
    rel_path = row.filename_lr
    full_path = os.path.join(DATA_DIR, rel_path)
    
    # Auto-download any missing waveform files
    if not os.path.exists(full_path + '.hea'):
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        wfdb.dl_files('ptb-xl', DATA_DIR, [rel_path + '.hea', rel_path + '.dat'])
        
    # Read the signal array and calculate metrics
    signal, _ = wfdb.rdsamp(full_path)
    feats = extract_ecg_features(signal)
    
    X_list.append(feats)
    y_list.append(row['label'])

# 5. Export Tabular Dataset
df_features = pd.DataFrame(X_list, columns=feature_names)
df_features['target'] = y_list
df_features.to_csv('ecg_features_extracted.csv', index=False)

print("\nFeature extraction complete!")
print(f"Dataset saved to 'ecg_features_extracted.csv' with shape: {df_features.shape}")