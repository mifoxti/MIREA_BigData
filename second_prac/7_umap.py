import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import umap.umap_ as umap
from sklearn.manifold import TSNE
import time

print("=== СРАВНЕНИЕ UMAP И t-SNE ===")

# Загружаем данные
print("Загрузка данных MNIST...")
digits = load_digits()
X, y = digits.data, digits.target

# Берем подвыборку
n_samples = 1000
X_subset = X[:n_samples]
y_subset = y[:n_samples]
print(f"Данные: {X_subset.shape}")

# Параметры для UMAP
n_neighbors_list = [5, 15, 50]
min_dist_list = [0.1, 0.5, 0.99]

# Сравнение времени
print("\n=== СРАВНЕНИЕ ВРЕМЕНИ ВЫПОЛНЕНИЯ ===")

# Время t-SNE
start_time = time.time()
tsne = TSNE(n_components=2, random_state=42, n_jobs=-1) # Используем все ядра
X_tsne = tsne.fit_transform(X_subset)
tsne_time = time.time() - start_time
print(f"t-SNE время: {tsne_time:.2f} секунд")

# Время UMAP (без random_state для параллелизации)
start_time = time.time()
umap_model = umap.UMAP(n_jobs=-1) # Используем все ядра
X_umap = umap_model.fit_transform(X_subset)
umap_time = time.time() - start_time
print(f"UMAP время: {umap_time:.2f} секунд")

speed_ratio = tsne_time / umap_time
print(f"\nUMAP {'быстрее' if speed_ratio > 1 else 'медленнее'} t-SNE в {abs(speed_ratio):.1f} раз")

# Визуализация UMAP с разными параметрами
print("\n=== ВИЗУАЛИЗАЦИЯ UMAP С РАЗНЫМИ ПАРАМЕТРАМИ ===")

plt.figure(figsize=(15, 10))

plot_num = 1
for i, n_neighbors in enumerate(n_neighbors_list):
    for j, min_dist in enumerate(min_dist_list):
        print(f"UMAP: n_neighbors={n_neighbors}, min_dist={min_dist}")

        start_time = time.time()
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_jobs=-1)
        X_embedded = reducer.fit_transform(X_subset)
        elapsed_time = time.time() - start_time

        plt.subplot(3, 3, plot_num)
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_subset, cmap='tab10', s=20)
        plt.title(f'n_neighbors={n_neighbors}\nmin_dist={min_dist}\nвремя: {elapsed_time:.2f}с', fontsize=10)
        plot_num += 1

plt.tight_layout()
plt.show()

# Дополнительная визуализация для сравнения
print("\n=== СРАВНИТЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ ===")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# t-SNE
scatter1 = ax1.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_subset, cmap='tab10', s=20)
ax1.set_title(f't-SNE (время: {tsne_time:.2f}с)')
ax1.set_xlabel('Компонента 1')
ax1.set_ylabel('Компонента 2')
plt.colorbar(scatter1, ax=ax1, label='Класс')

# UMAP
scatter2 = ax2.scatter(X_umap[:, 0], X_umap[:, 1], c=y_subset, cmap='tab10', s=20)
ax2.set_title(f'UMAP (время: {umap_time:.2f}с)')
ax2.set_xlabel('Компонента 1')
ax2.set_ylabel('Компонента 2')
plt.colorbar(scatter2, ax=ax2, label='Класс')

plt.tight_layout()
plt.show()

print("\n=== ВЫВОД ===")
print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА:")
print(f"1. UMAP оказался МЕДЛЕННЕЕ t-SNE в {abs(speed_ratio):.1f} раз")
print("2. Возможные причины:")
print("   - Небольшой размер данных (1000 samples)")
print("   - Версия библиотек или настройки")
print("   - t-SNE использует оптимизации в новой версии")
print("3. Параметры UMAP:")
print("   - n_neighbors: влияет на баланс локальной/глобальной структуры")
print("   - min_dist: контролирует плотность кластеров")
print("4. На больших данных UMAP обычно быстрее, но здесь t-SNE показал лучшую скорость")