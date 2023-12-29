import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# CSV 파일 읽기
data = pd.read_csv('데이터/결과데이터.csv', encoding='utf-8')

# '학교' 열을 index로 설정
data.set_index('학교', inplace=True)

# 주성분 분석 (PCA)을 사용하여 차원을 95로 축소
pca = PCA(n_components=95)
data_pca = pd.DataFrame(pca.fit_transform(data), index=data.index)

# K-means 클러스터링 수행
k_values = range(2, 20)  # Try different numbers of clusters
inertia_values = []
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k)
    clusters = kmeans.fit_predict(data_pca)
    
    inertia_values.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(data_pca, clusters))

# Plot elbow method and silhouette score
plt.figure(figsize=(12, 6))

# Plot Inertia (Within-cluster Sum of Squares)
plt.subplot(1, 2, 1)
plt.plot(k_values, inertia_values, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia (Within-cluster Sum of Squares)')

# Plot Silhouette Score
plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette Score vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')

plt.tight_layout()
plt.show()

# Choose the optimal number of clusters based on the elbow method
optimal_k_elbow = 3  # Adjust this based on the elbow point in the graph

# Choose the optimal number of clusters based on silhouette score
optimal_k_silhouette = k_values[silhouette_scores.index(max(silhouette_scores))]

# Choose the common optimal K
optimal_k = optimal_k_elbow if optimal_k_silhouette <= 1 else optimal_k_silhouette

# K-means clustering with the optimal number of clusters
kmeans_optimal = KMeans(n_clusters=optimal_k)
clusters_optimal = kmeans_optimal.fit_predict(data_pca)

# Add cluster labels to the dataframe
data['Cluster'] = clusters_optimal

# Visualize clusters using PCA (2D plot)
data_pca['Cluster'] = clusters_optimal

plt.figure(figsize=(12, 8))
for i in range(optimal_k):
    cluster_data = data_pca[data_pca['Cluster'] == i]
    plt.scatter(cluster_data[0], cluster_data[1], label=f'Cluster {i + 1}')

plt.title('K-means Clustering Results (2D PCA) - Optimal K')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# Print cluster information
for i in range(data['Cluster'].nunique()):
    cluster_data = data[data['Cluster'] == i]
    print(f'그룹 {i + 1}에 속한 학교들:')
    print(cluster_data.index)
    print("\n")

# Print common subjects in each cluster
for i in range(data['Cluster'].nunique()):
    cluster_data = data[data['Cluster'] == i].drop('Cluster', axis=1)
    common_subjects = cluster_data.columns[cluster_data.all(axis=0)]
    
    print(f'그룹 {i + 1}에 속한 학교들에서 공통으로 개설된 교과목:')
    print(common_subjects)
    print("\n")
