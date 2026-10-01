import os
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wfdb

# ---------------------------------------------------------
# 1. Configuration & Data Directory Setup
# ---------------------------------------------------------
DATA_DIR = './data/ptb-xl/'
SAMPLING_RATE = 100  # Use 100 Hz for fast local processing on Mac M4

os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------
# 2. Download PTB-XL Metadata (If not already present)
# ---------------------------------------------------------
metadata_url = 'https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv'
scp_url = 'https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv'

if not os.path.exists(os.path.join(DATA_DIR, 'ptbxl_database.csv')):
    print("Downloading PTB-XL metadata files...")
    df_meta = pd.read_csv(metadata_url, index_col='ecg_id')
    df_scp = pd.read_csv(scp_url, index_col=0)
    df_meta.to_csv(os.path.join(DATA_DIR, 'ptbxl_database.csv'))
    df_scp.to_csv(os.path.join(DATA_DIR, 'scp_statements.csv'))
else:
    print("Metadata files found locally.")
    df_meta = pd.read_csv(os.path.join(DATA_DIR, 'ptbxl_database.csv'), index_col='ecg_id')
    df_scp = pd.read_csv(os.path.join(DATA_DIR, 'scp_statements.csv'), index_col=0)

# ---------------------------------------------------------
# 3. Parse SCP Codes to Diagnostic Superclasses
# ---------------------------------------------------------
# Parse SCP code dictionary strings into actual dicts
df_meta.scp_codes = df_meta.scp_codes.apply(lambda x: ast.literal_eval(x))

# Filter diagnostic statements only
diagnostic_scp = df_scp[df_scp.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    """Maps detailed SCP diagnostic codes to 5 main diagnostic superclasses:
    NORM (Normal), MI (Myocardial Infarction), STTC (ST/T Change), 
    CD (Conduction Disturbance), HYP (Hypertrophy)
    """
    tmp = []
    for key in y_dic.keys():
        if key in diagnostic_scp.index:
            tmp.append(diagnostic_scp.loc[key].diagnostic_class)
    return list(set(tmp))

df_meta['diagnostic_superclass'] = df_meta.scp_codes.apply(aggregate_diagnostic)

# ---------------------------------------------------------
# 4. Class Distribution Visual (Save for Report EDA)
# ---------------------------------------------------------
# Explode list of classes to count occurrences across records
all_classes = df_meta['diagnostic_superclass'].explode().value_counts()

print("\n--- Diagnostic Class Counts ---")
print(all_classes)

plt.figure(figsize=(8, 4))
all_classes.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('PTB-XL Diagnostic Superclass Distribution')
plt.xlabel('Diagnostic Class')
plt.ylabel('Number of Records')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('eda_class_distribution.png', dpi=300)
plt.show()

# ---------------------------------------------------------
# 5. Download and Load Sample Waveform
# ---------------------------------------------------------
# Download first record (ecg_id = 1) to test wfdb reader
sample_record_path = df_meta.iloc[0].filename_lr
record_dir = os.path.dirname(os.path.join(DATA_DIR, sample_record_path))
os.makedirs(record_dir, exist_ok=True)

print(f"\nFetching sample record: {sample_record_path}")
wfdb.dl_files('ptb-xl', DATA_DIR, [sample_record_path + '.hea', sample_record_path + '.dat'])

# Read signal using wfdb
signal, meta = wfdb.rdsamp(os.path.join(DATA_DIR, sample_record_path))

# ---------------------------------------------------------
# 6. Plot 12-Lead Waveform (Save for Report EDA)
# ---------------------------------------------------------
lead_names = meta['sig_name']
time_axis = np.arange(signal.shape[0]) / SAMPLING_RATE

plt.figure(figsize=(14, 10))
for i in range(12):
    plt.subplot(6, 2, i + 1)
    plt.plot(time_axis, signal[:, i], color='darkblue', linewidth=0.8)
    plt.ylabel(lead_names[i])
    plt.grid(True, linestyle='--', alpha=0.5)
    if i < 10:
        plt.xticks([])
    else:
        plt.xlabel('Time (seconds)')

plt.suptitle('Sample 12-Lead ECG Signal (Patient ID: 1)', fontsize=14)
plt.tight_layout()
plt.savefig('eda_sample_12lead_ecg.png', dpi=300)
plt.show()