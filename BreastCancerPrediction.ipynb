import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV as GScv
from sklearn.ensemble import RandomForestClassifier as RFC, GradientBoostingClassifier as GBC
from sklearn.metrics import classification_report, roc_auc_score as AUC, roc_curve as ROC
from sklearn.preprocessing import LabelEncoder as LE, StandardScaler as SS

data_path = 'BreastCancer.csv'
data = pd.read_csv("C:/Users/gokul/Downloads/gova/New folder/BreastCancer.csv")

data.rename(columns=lambda x: x.strip(), inplace=True)

if 'diagnosis' in data.columns:
    data.rename(columns={'diagnosis': 'Diagnosis'}, inplace=True)

if 'Diagnosis' not in data.columns:
    print("Available Columns:", data.columns)
    raise KeyError("The 'Diagnosis' column is not found in the dataset. Please verify the dataset structure.")

le = LE()
data['Diagnosis'] = le.fit_transform(data['Diagnosis'])

plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), cmap="coolwarm", annot=False)
plt.show()

features = data.drop(['id', 'Diagnosis', 'Unnamed: 32'], axis=1, errors='ignore')
target = data['Diagnosis']

scaler = SS()
features_scaled = scaler.fit_transform(features)

features_train, features_test, target_train, target_test = train_test_split(
    features_scaled, target, test_size=0.2, random_state=42, stratify=target
)

rf = RFC(random_state=42)
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}
rfg = GScv(rf, rf_params, cv=5, scoring='f1', verbose=1)
rfg.fit(features_train, target_train)

best_rf = rfg.best_estimator_
rf_preds = best_rf.predict(features_test)
print(classification_report(target_test, rf_preds))

gb = GBC(random_state=42)
gb_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
}
gbg = GScv(gb, gb_params, cv=5, scoring='f1', verbose=1)
gbg.fit(features_train, target_train)

best_gb = gbg.best_estimator_
gb_preds = best_gb.predict(features_test)
print(classification_report(target_test, gb_preds))

rfauc = AUC(target_test, best_rf.predict_proba(features_test)[:, 1])
gbauc = AUC(target_test, best_gb.predict_proba(features_test)[:, 1])

rffpr, rftpr, _ = ROC(target_test, best_rf.predict_proba(features_test)[:, 1])
gbfpr, gbtpr, _ = ROC(target_test, best_gb.predict_proba(features_test)[:, 1])

plt.figure(figsize=(10, 6))
plt.plot(rffpr, rftpr, label=f"Random Forest (AUC = {rfauc:.2f})")
plt.plot(gbfpr, gbtpr, label=f"Gradient Boosting (AUC = {gbauc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.legend()
plt.show()
