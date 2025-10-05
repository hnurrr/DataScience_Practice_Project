import os
import warnings
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Suppress parallelization warnings
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.cluster._kmeans')

# Load dataset
data = pd.read_csv('dava.csv')
print("Dataset preview:")
print(data.head())

# --- BASIC INFORMATION ---
print("\n=== Dataset Info ===")
print(data.info())

print("\n=== Statistical Summary ===")
print(data.describe())

print("\n=== Missing Values ===")
print(data.isnull().sum())

# --- FEATURE SELECTION ---
features = ['Case Duration (Days)', 'Number of Witnesses', 'Legal Fees (USD)',
            'Number of Evidence Items', 'Severity']
X = data[features].copy()

print(f"\nSelected features for clustering: {features}")
print(f"Data shape: {X.shape}")

# --- DATA STANDARDIZATION ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- ELBOW METHOD ---
inertias = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, marker='o', linewidth=2)
plt.title('Elbow Method - Determining Optimal Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (WCSS)')
plt.grid(alpha=0.3)
plt.show()

print("\nInspect the Elbow plot to choose the optimal k (cluster count).")

# --- K-MEANS CLUSTERING ---
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

data_clustered = data.copy()
data_clustered['Cluster'] = cluster_labels

print(f"\nK-Means completed with {optimal_k} clusters.")
print(data_clustered['Cluster'].value_counts().sort_index())

# --- CLUSTER CENTERS (ORIGINAL SCALE) ---
cluster_centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(cluster_centers_original, columns=features)
centers_df.index.name = 'Cluster'

print("\nCluster centers (original scale):")
print(centers_df.round(2))

# --- SCATTER PLOTS FOR CLUSTERS ---
feature_pairs = [
    ('Case Duration (Days)', 'Legal Fees (USD)'),
    ('Number of Witnesses', 'Number of Evidence Items'),
    ('Case Duration (Days)', 'Severity'),
    ('Legal Fees (USD)', 'Severity'),
    ('Number of Witnesses', 'Legal Fees (USD)'),
    ('Number of Evidence Items', 'Case Duration (Days)')
]

colors = ['red', 'blue', 'green']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('K-Means Clustering Results - Feature Pairs', fontsize=16)

for i, (x_feature, y_feature) in enumerate(feature_pairs):
    ax = axes[i // 3, i % 3]
    for cluster in range(optimal_k):
        cluster_data = data_clustered[data_clustered['Cluster'] == cluster]
        ax.scatter(cluster_data[x_feature], cluster_data[y_feature],
                   c=colors[cluster], label=f'Cluster {cluster}', alpha=0.7, s=50)
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# --- CLUSTER-WISE STATISTICS ---
print("\n=== CLUSTER ANALYSIS ===")
for cluster in range(optimal_k):
    cluster_data = data_clustered[data_clustered['Cluster'] == cluster]
    print(f"\n--- CLUSTER {cluster} ---")
    print(f"Sample count: {len(cluster_data)}")
    print("Mean feature values:")
    for feature in features:
        print(f"  {feature}: {cluster_data[feature].mean():.2f}")

    outcome_counts = cluster_data['Outcome'].value_counts()
    print(f"Outcome distribution -> Unfavorable: {outcome_counts.get(0, 0)}, Favorable: {outcome_counts.get(1, 0)}")

# --- FEATURE DISTRIBUTION BY CLUSTER ---
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Feature Distributions by Cluster', fontsize=16)

for i, feature in enumerate(features):
    ax = axes[i // 3, i % 3]
    cluster_values = [data_clustered[data_clustered['Cluster'] == c][feature] for c in range(optimal_k)]
    ax.boxplot(cluster_values, tick_labels=[f'Cluster {c}' for c in range(optimal_k)])
    ax.set_title(feature)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# --- CLUSTER HEATMAP ---
plt.figure(figsize=(10, 6))
plt.imshow(cluster_centers_original.T, cmap='viridis', aspect='auto')
plt.colorbar(label='Value')
plt.yticks(range(len(features)), features)
plt.xticks(range(optimal_k), [f'Cluster {i}' for i in range(optimal_k)])
plt.title('Cluster Centers Heatmap')

for i in range(len(features)):
    for j in range(optimal_k):
        plt.text(j, i, f'{cluster_centers_original[j, i]:.1f}',
                 ha='center', va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.show()

print("\n=== SUMMARY ===")
print("1. Clustering analysis completed successfully.")
print("2. Each cluster represents a distinct case profile.")
print("3. The results may help identify similar case patterns.")
print("4. Visuals can be used for further interpretation.")


