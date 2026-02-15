import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt 

# 1. Load and Prepare Data
data = pd.read_csv("robotics.csv")

# Select features that our EDA showed are important
# We include Torque, RPM, Tool wear, and Temperature difference
data['Temp_Diff'] = data['Process temperature [K]'] - data['Air temperature [K]']
features = ['Torque [Nm]', 'Rotational speed [rpm]', 'Tool wear [min]', 'Temp_Diff']
X = data[features]
y = data['Machine failure']

# 2. Train/Test Split
# We use stratify because failures are rare (imbalanced data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y)

# 3. Scaling (Crucial for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Logistic Regression Model
model = LogisticRegression(class_weight='balanced') # 'balanced' helps with the rare failures
model.fit(X_train_scaled, y_train)

# 5. Predictions and Evaluation
y_pred = model.predict(X_test_scaled)

print("--- Logistic Regression Baseline Results ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. Confusion Matrix Visualization
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix: Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# 7. Feature Importance (Weights)
importance = pd.DataFrame({'Feature': features, 'Weight': model.coef_[0]})
importance = importance.sort_values(by='Weight', ascending=False)
print("\n--- Feature Importance (Coefficients) ---")
print(importance)