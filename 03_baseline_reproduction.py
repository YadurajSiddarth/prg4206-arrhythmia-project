import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score, classification_report

# 1. Load the locally extracted subset dataset
print("Loading dataset...")
df = pd.read_csv('ecg_features_extracted.csv')
X = df.drop(columns=['target'])
y = df['target']

# 2. Encode categorical diagnostic labels (NORM, MI, STTC, CD, HYP) into integers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# 3. Split data (80% Training, 20% Testing) using stratification to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 4. Initialize and train the baseline Random Forest (Mimicking Strodthoff et al. baseline)
print("Training baseline Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 5. Predict probabilities (for AUC) and discrete classes (for F1/Accuracy)
y_pred_proba = rf_model.predict_proba(X_test)
y_pred = rf_model.predict(X_test)

# 6. Calculate benchmark evaluation metrics
macro_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
macro_f1 = f1_score(y_test, y_pred, average='macro')

print("\n--- Baseline Reproduction Results ---")
print(f"Macro ROC-AUC:  {macro_auc:.4f} (Original Paper Target: ~0.837)")
print(f"Macro F1-Score: {macro_f1:.4f}\n")
print("Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))