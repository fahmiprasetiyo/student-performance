import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the dataset
try:
    df = pd.read_csv('student-mat.csv', sep=';')
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# 1. Select features for clustering
features = ['studytime', 'absences']
X = df[features]

# 2. Scale the features
# K-Means is distance-based, so scaling is important
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Determine the optimal number of clusters using the Elbow Method
wcss = []  # Within-Cluster Sum of Squares
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, marker='o', linestyle='--')
plt.title('Elbow Method untuk Menentukan Jumlah Cluster Optimal', fontsize=16)
plt.xlabel('Jumlah Cluster (K)', fontsize=12)
plt.ylabel('WCSS (Inertia)', fontsize=12)
plt.xticks(k_range)
plt.grid(True)
plt.savefig('elbow_method.png')
plt.close()

# 4. Apply K-Means with the optimal K
# From the elbow plot, K=4 seems like a good choice.
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
# Fit the model and predict clusters
df['cluster'] = kmeans.fit_predict(X_scaled)

# 5. Visualize the clusters
plt.figure(figsize=(12, 8))
scatter = sns.scatterplot(
    data=df,
    x='absences',
    y='studytime',
    hue='cluster',
    palette='bright',
    s=100,
    alpha=0.7,
    edgecolor='k'
)

plt.title('Segmentasi Siswa Berdasarkan Waktu Belajar dan Absensi', fontsize=16)
plt.xlabel('Jumlah Absensi', fontsize=12)
plt.ylabel('Waktu Belajar (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)', fontsize=12)
plt.legend(title='Cluster')
plt.yticks([1, 2, 3, 4])
plt.grid(True)
plt.savefig('student_clusters.png')
plt.close()

# Print cluster centers to help with interpretation
print("Cluster centers (in original scale):")

cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_info = pd.DataFrame(cluster_centers, columns=features)
print(cluster_info)


print("\nClustering analysis complete. elbow_method.png and student_clusters.png have been generated.")
