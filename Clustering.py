import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
try:
    df = pd.read_csv('student-mat.csv', sep=';')
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# --- Clustering ---

# 1. Select features for clustering
features = ['studytime', 'absences']
X = df[features]

# 2. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Determine the optimal number of clusters using the Elbow Method
wcss = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, marker='o', linestyle='--')
plt.title('Metode Elbow untuk Menentukan Jumlah Cluster Optimal', fontsize=16)
plt.xlabel('Jumlah Cluster (K)', fontsize=12)
plt.ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12)
plt.xticks(k_range)
plt.grid(True)
plt.savefig('elbow_method.png')
plt.close()

# 4. Perform K-Means Clustering with the optimal K
# Based on the elbow plot, K=4 seems like a reasonable choice.
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add the cluster labels to the original DataFrame
df['cluster'] = cluster_labels

# 5. Visualize the clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(x='studytime', y='absences', hue='cluster', data=df, palette='viridis', s=100, alpha=0.8, legend='full')
plt.title('Segmentasi Siswa Berdasarkan Waktu Belajar dan Absensi', fontsize=16)
plt.xlabel('Waktu Belajar (1: <2 jam, 2: 2-5 jam, 3: 5-10 jam, 4: >10 jam)', fontsize=12)
plt.ylabel('Jumlah Absensi', fontsize=12)
plt.legend(title='Cluster')
plt.grid(True)
plt.savefig('student_clusters.png')
plt.close()

# 6. Analyze the clusters
print("--- Analisis Karakteristik Cluster ---")
cluster_analysis = df.groupby('cluster')[features].mean().round(2)
print(cluster_analysis)

print("\nClustering visualizations have been generated and saved as PNG files.")