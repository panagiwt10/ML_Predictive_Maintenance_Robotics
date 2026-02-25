import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


data = pd.read_csv("robotics.csv")
print("columns of dataset:", data.columns.tolist())

failures = data[data['Machine failure']== 1 ] 
success =  data[data['Machine failure'] == 0 ]

print(f"Failures: {len(failures)}")
print(f"Successes: {len(success)}")


failure_modes = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
failure_counts = data[failure_modes].sum().sort_values(ascending=False)
print(failure_counts)

# XGBoost algorythm 

data['Temp_Diff'] = data['Process temperature [K]'] - data['Air temperature [K]']
features = ['Torque [Nm]', 'Rotational speed [rpm]', 'Tool wear [min]', 'Temp_Diff']

#Remove brackets and spaces from column names
data.columns = data.columns.str.replace('[', '', regex=False).str.replace(']', '', regex=False).str.replace(' ', '_', regex=False)
features = ['Torque_Nm', 'Rotational_speed_rpm', 'Tool_wear_min', 'Temp_Diff']

X = data[features]
y = data['Machine_failure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Calculate Scale Weight for Imbalanced Data
# ratio = negative_cases / positive_cases
ratio = float(y_train.value_counts()[0] / y_train.value_counts()[1])

xbgm_model = XGBClassifier(
    n_estimators=100, 
    max_depth=4, 
    learning_rate=0.1, 
    scale_pos_weight=ratio , # Critical for imbalanced failure data
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

xbgm_model.fit(X_train, y_train)

# Predictions
y_pred_xgb = xbgm_model.predict(X_test)

# Evaluation
print("--- XGBoost Results ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred_xgb):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))


import joblib 
model_filename = "robotics_xgb_model.pkl"
joblib.dump(xbgm_model, model_filename)

print(f"saved: {model_filename}")