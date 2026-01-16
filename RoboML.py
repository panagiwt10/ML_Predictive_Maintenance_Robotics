import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("Robo_metrics.csv")
print(df.head())

# Διαχωρισμός: Κρατάμε τη στήλη 'fault' χωριστά
df_fault = df['fault']

df = df.drop(columns=["fault"], errors="ignore")
print(df.head())

stats = df.describe().T  # describe() μας δίνει πολλά στατιστικά, το .T τα μετατρέπει σε γραμμές
print("Statistics:",stats)


# ------------------------------------------------
# Plot 1: Ιστογράμματα για όλες τις αριθμητικές μετρικές
# ------------------------------------------------

n_rows = 3
n_cols = 3
plt.figure(figsize=(15, 15))

for i, col in enumerate(df):
    plt.subplot(n_rows, n_cols, i + 1)
    # Χρησιμοποιούμε histplot με kde (εκτίμηση πυκνότητας πυρήνα) για ομαλή γραμμή
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Κατανομή: {col}', fontsize=12)
    plt.xlabel(col, fontsize=10)
    plt.ylabel('Πλήθος', fontsize=10)
    plt.tight_layout()

plt.suptitle('Ιστογράμματα Κατανομής για όλες τις Αριθμητικές Μετρικές', fontsize=16, y=1.02)
plt.savefig('numerical_histograms.png')
plt.close()

# ------------------------------------------------
# Plot 2: Correlation Matrix Heatmap
# ------------------------------------------------

correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,          # Εμφάνιση των τιμών συσχέτισης
    fmt=".2f",           # Μορφοποίηση σε 2 δεκαδικά ψηφία
    cmap='coolwarm',     # Χρωματικός χάρτης (Cool-Warm)
    linewidths=.5,       # Γραμμές μεταξύ των κελιών
    cbar_kws={'label': 'Συντελεστής Συσχέτισης'} # Ετικέτα για τη χρωματική μπάρα
)
plt.title('Διάγραμμα Θερμότητας (Heatmap) Πινάκων Συσχέτισης', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()

# ------------------------------------------------
# Min-max Norm 
# ------------------------------------------------


scaler = StandardScaler()
normalized_data = scaler.fit_transform(df)
print("Normalized Data (Min-Max Scaling):")
print(normalized_data)

# ------------------------------------------------
# BoxPlot : std scaler Norm 
# ------------------------------------------------

plt.figure(figsize=(12,6))
plt.boxplot(normalized_data, labels=df.columns, vert=False)
plt.xticks(rotation=45)
plt.title("Standardized feature distributions")
plt.show()


# ------------------------------------------------
# PCA  
# ------------------------------------------------

# 4. Apply PCA and transform the data (ΜΟΝΟ στα αριθμητικά δεδομένα)
pca = PCA(n_components=2)
principal_comp = pca.fit_transform(normalized_data)

df_pca = pd.DataFrame(data = principal_comp , columns= ['PC1', 'PC2'])
df_pca['fault'] = df_fault # <-- Η 'fault' προστίθεται εδώ ξανά!

# Scatter Plot (Χρήση της 'fault' για το hue)
plt.figure(figsize=(10,8))
sns.scatterplot(
    x='PC1', 
    y='PC2', 
    hue ='fault', 
    data = df_pca, 
    palette='Spectral',
    s=50,
    alpha=0.7
    )
plt.show()

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total explained variance:", pca.explained_variance_ratio_.sum())


selected_features = df[['rms', 'kurtosis', 'skewness', 'crest']]

le = LabelEncoder()
y = le.fit_transform(df_fault)



# train / test split 
X = normalized_data
X_train , X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.25,random_state=42, stratify=y
)

#Baseline classifier - RandomForest 

clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))


import joblib 

#1. οπτικοποιηση confusion Matrix 
plt.figure(figsize=(12,9))
sns.heatmap(confusion_matrix(y_test, y_pred), annot = True, fmt = 'd', cmap = 'Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Πίνακας Συγχύσεως (Confusion Matrix) - Πού μπερδεύεται το ρομπότ;')
plt.xlabel('Πρόβλεψη Μοντέλου')
plt.ylabel('Πραγματική Κατάσταση')
plt.tight_layout()
plt.savefig('confusion_matrix_heatmap.png')
plt.show()

# 2. Οπτικοποίηση Σημαντικότητας Χαρακτηριστικών
importances = clf.feature_importances_
feature_names = df.columns # Αφού έχεις αφαιρέσει το 'fault'
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)


plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Ποιες μετρικές "κοιτάζει" περισσότερο το ρομπότ;')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# 3. Αποθήκευση Μοντέλου και Scaler
joblib.dump(clf, 'robotic_maintenance_model.pkl')
joblib.dump(scaler, 'data_scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("Το μοντέλο και οι μετασχηματιστές saved")