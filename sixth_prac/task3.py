import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# Загрузка подготовленных данных
data = np.loadtxt('prepared_data.csv', delimiter=',')

# Выборка подмножества для иерархической кластеризации (чтобы избежать большого дендрограммы)
sample_size = min(100, len(data))  # Использовать до 100 образцов для визуализации
indices = np.random.choice(len(data), sample_size, replace=False)
data_sample = data[indices]

# Выполнение иерархической кластеризации
Z = linkage(data_sample, method='ward')

# Построение дендрограммы
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title('Дендрограмма иерархической кластеризации (Выборка)')
plt.xlabel('Индекс образца')
plt.ylabel('Расстояние')
plt.savefig('hierarchical_dendrogram.png')
plt.show()

print("Иерархическая кластеризация завершена. Дендрограмма сохранена.")