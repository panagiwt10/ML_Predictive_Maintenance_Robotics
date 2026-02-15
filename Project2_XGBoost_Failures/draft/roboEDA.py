import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Φόρτωση δεδομένων
data = pd.read_csv("robotics.csv")

# Εκτύπωση στηλών για επιβεβαίωση
print("Στήλες στο dataset:", data.columns.tolist())

# Φιλτράρουμε τις αστοχίες και τις επιτυχίες
failures = data[data['Machine failure'] == 1]
success = data[data['Machine failure'] == 0]

# Εκτύπωση πλήθους χωρίς ολόκληρο τον πίνακα
print(f"Συνολικές Αστοχίες (Failures): {len(failures)}")
print(f"Συνολικές Επιτυχίες (Successes): {len(success)}")

# Scatter plot μόνο για τις αστοχίες
plt.figure(figsize=(10, 6))
sns.scatterplot(data=failures, x="Rotational speed [rpm]", y="Torque [Nm]", hue="Type", palette="hot")
plt.title("Αστοχίες: Ταχύτητα Περιστροφής vs Ροπή", fontsize=14)
plt.savefig("failures_scatter.png")

# Scatter plot μόνο για τις επιτυχίες
plt.figure(figsize=(10, 6))
sns.scatterplot(data=success, x="Rotational speed [rpm]", y="Torque [Nm]", hue="Type", alpha=0.3)
plt.title("Επιτυχίες: Ταχύτητα Περιστροφής vs Ροπή", fontsize=14)
plt.savefig("success_scatter.png")

# Σύγκριση Ροπής για Failures vs Non-Failures με Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x="Machine failure", y="Torque [Nm]", data=data, palette="coolwarm")
plt.title("Κατανομή Ροπής ανά Κατάσταση Μηχανής", fontsize=14)
plt.savefig("torque_boxplot.png")

# Φιλτράρουμε μόνο τις αστοχίες
failures = data[data['Machine failure'] == 1]

# Παίρνουμε τους μοναδικούς τύπους μηχανημάτων (L, M, H)
machine_types = failures['Type'].unique()

# Loop για να εκτυπώσουμε κάθε τύπο σε ξεχωριστό γράφημα
for m_type in machine_types:
    # Φιλτράρουμε τις αστοχίες για τον συγκεκριμένο τύπο
    subset = failures[failures['Type'] == m_type]
    
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=subset, 
        x="Rotational speed [rpm]", 
        y="Torque [Nm]", 
        color="red", 
        s=100, 
        edgecolor="black"
    )
    
    plt.title(f"Αστοχίες για Μηχανήματα Τύπου: {m_type}", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # TWF (Tool Wear Failure): Η μηχανή χάλασε λόγω υπερβολικής φθοράς του εργαλείου.
    # HDF (Heat Dissipation Failure): Βλάβη λόγω κακής απαγωγής θερμότητας (υπερθέρμανση).
    # PWF (Power Failure): Βλάβη λόγω ισχύος (σχέση ροπής και ταχύτητας).
    # OSF (Overstrain Failure): Βλάβη λόγω υπερβολικού φορτίου/καταπόνησης.
    # RNF (Random Failures): Τυχαίες βλάβες που δεν προβλέπονται εύκολα από τις μετρήσεις. 

    # Επιλέγουμε τις στήλες των βλαβών
failure_modes = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
# Αθροίζουμε τις τιμές (αφού είναι 0 και 1)
failure_counts = data[failure_modes].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=failure_counts.index, y=failure_counts.values, palette="viridis")
plt.title("Συχνότητα Τύπων Βλάβης", fontsize=16)
plt.ylabel("Αριθμός Περιστατικών")
plt.show()

# Προσθέτουμε και τη στήλη Machine failure στην ανάλυση
cols_to_check = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']

plt.figure(figsize=(8, 6))
sns.heatmap(data[cols_to_check].corr(), annot=True, cmap="Reds", fmt=".2f")
plt.title("Συσχέτιση Αιτιών με τη Γενική Βλάβη")
plt.show()

# Υπολογίζουμε πόσες βλάβες έχει κάθε σειρά
data['Total_Failures'] = data[failure_modes].sum(axis=1)

# Δείχνουμε μόνο τις περιπτώσεις που έχουν πάνω από 1 αιτία
multi_failure = data[data['Total_Failures'] > 1]
print(f"Βρέθηκαν {len(multi_failure)} περιπτώσεις με πολλαπλές αιτίες βλάβης.")


features = ['Torque [Nm]', 'Rotational speed [rpm]', 'Tool wear [min]', 'Process temperature [K]']
X = data[features]

# Scaling (Απαραίτητο για τον K-means)
# Φέρνουμε όλες τις τιμές στην ίδια κλίμακα (π.χ. mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# clustering with Kmeans 
"""
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Εφαρμογή του K-means (επιλέγουμε π.χ. K=4 βάσει του προηγούμενου πειράματος)
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Οπτικοποίηση των Clusters σε σχέση με Ροπή και Στροφές
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x="Rotational speed [rpm]", y="Torque [Nm]", hue="Cluster", palette="viridis", alpha=0.6)
plt.title("Ομαδοποίηση Μηχανημάτων (K-means Clusters)")
plt.show()

# Στατιστική Ανάλυση: Πόσες βλάβες έχουμε σε κάθε Cluster;
cluster_summary = data.groupby('Cluster')['Machine failure'].agg(['count', 'sum'])
cluster_summary.columns = ['Συνολικά Σημεία', 'Αριθμός Βλαβών']
cluster_summary['Ποσοστό Βλάβης (%)'] = (cluster_summary['Αριθμός Βλαβών'] / cluster_summary['Συνολικά Σημεία'] * 100).round(2)

print("\n--- Ανάλυση Clusters ---")
print(cluster_summary)

# (Προαιρετικό) Δες ποια είναι τα χαρακτηριστικά του κάθε Cluster
print("\n--- Μέσοι Όροι ανά Cluster ---")
print(data.groupby('Cluster')[features].mean().round(2))
"""

