import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster

# Загрузка подготовленных данных
data = np.loadtxt('prepared_data.csv', delimiter=',')

# Снижение размерности с помощью t-SNE
tsne = TSNE(n_components=2, random_state=42)
data_2d = tsne.fit_transform(data)

plt.figure(figsize=(16, 6))

# Визуализация для K-means
plt.subplot(1, 3, 1)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels_kmeans = kmeans.fit_predict(data)
scatter1 = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels_kmeans, cmap='viridis', alpha=0.6)
plt.colorbar(scatter1)
plt.title('Визуализация t-SNE кластеров K-means')
plt.xlabel('Компонента t-SNE 1')
plt.ylabel('Компонента t-SNE 2')

# Визуализация для иерархической кластеризации
plt.subplot(1, 3, 2)
Z = linkage(data, method='ward')
labels_hier = fcluster(Z, t=optimal_k, criterion='maxclust')
scatter2 = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels_hier, cmap='plasma', alpha=0.6)
plt.colorbar(scatter2)
plt.title('Визуализация t-SNE кластеров иерархической')
plt.xlabel('Компонента t-SNE 1')
plt.ylabel('Компонента t-SNE 2')

# Визуализация для DBSCAN
plt.subplot(1, 3, 3)
eps = 2.0
min_samples = 5
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels_dbscan = dbscan.fit_predict(data)
scatter3 = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels_dbscan, cmap='inferno', alpha=0.6)
plt.colorbar(scatter3)
plt.title('Визуализация t-SNE кластеров DBSCAN')
plt.xlabel('Компонента t-SNE 1')
plt.ylabel('Компонента t-SNE 2')

plt.tight_layout()
plt.savefig('tsne_visualization.png')
plt.show()

print("Визуализация t-SNE для K-means, иерархической и DBSCAN завершена и сохранена.")