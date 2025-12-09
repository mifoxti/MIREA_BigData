import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Загрузка подготовленных данных
data = np.loadtxt('prepared_data.csv', delimiter=',')

# Построение графика k-distance для выбора eps
min_samples = 5
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(data)
distances, indices = neighbors_fit.kneighbors(data)
distances = np.sort(distances[:, min_samples-1], axis=0)

plt.figure(figsize=(8, 5))
plt.plot(distances)
plt.xlabel('Точки, отсортированные по расстоянию')
plt.ylabel(f'Расстояние до {min_samples}-го ближайшего соседа')
plt.title('График k-distance для выбора eps')
plt.grid(True)
plt.savefig('k_distance_plot.png')
plt.show()

# Выбор eps на основе графика (например, точка перегиба около 2.0)
eps = 2.0

# Выполнение кластеризации DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(data)

# Количество кластеров (исключая шум, который помечен -1)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"DBSCAN завершено с eps={eps}, min_samples={min_samples}")
print(f"Количество кластеров: {n_clusters}")
print(f"Количество точек шума: {n_noise}")

# Сохранение меток для визуализации
np.savetxt('dbscan_labels.csv', labels, delimiter=',')