import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Загрузка подготовленных данных
data = np.loadtxt('prepared_data.csv', delimiter=',')

# Определение оптимального k с использованием метода локтя и коэффициента силуэта
inertias = []
silhouettes = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(data, kmeans.labels_))

# График метода локтя
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertias, 'bx-')
plt.xlabel('k')
plt.ylabel('Инерция')
plt.title('Метод локтя')

# График коэффициентов силуэта
plt.subplot(1, 2, 2)
plt.plot(k_range, silhouettes, 'rx-')
plt.xlabel('k')
plt.ylabel('Коэффициент силуэта')
plt.title('Анализ силуэта')

plt.tight_layout()
plt.savefig('kmeans_elbow_silhouette.png')
plt.show()

# Выбор оптимального k (например, где локоть, скажем k=5)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels = kmeans.fit_predict(data)

print(f"Оптимальное k: {optimal_k}")
print("Кластеризация K-means завершена с анализом локтя и силуэта.")

# Сохранение меток для визуализации
np.savetxt('kmeans_labels.csv', labels, delimiter=',')