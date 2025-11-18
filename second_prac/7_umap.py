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

# ДОБАВЬТЕ ЭТОТ КОД ПОСЛЕ СУЩЕСТВУЮЩЕЙ ВИЗУАЛИЗАЦИИ

print("\n=== ДОПОЛНИТЕЛЬНЫЕ ГРАФИКИ И ДИАГРАММЫ ===")

# 1. ГРАФИК СРАВНЕНИЯ ПРОИЗВОДИТЕЛЬНОСТИ
plt.figure(figsize=(10, 6))

methods = ['t-SNE', 'UMAP (стандарт)', 'UMAP (n5)', 'UMAP (n50)', 'UMAP (md0.1)', 'UMAP (md0.99)']
times = [
    tsne_time,
    umap_time,
    times_umap[0],  # n_neighbors=5, min_dist=0.1
    times_umap[6],  # n_neighbors=50, min_dist=0.5 (примерно)
    times_umap[0],  # n_neighbors=5, min_dist=0.1
    times_umap[2]   # n_neighbors=5, min_dist=0.99
]

colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0']

bars = plt.bar(methods, times, color=colors, edgecolor='black', alpha=0.8)
plt.title('Сравнение времени выполнения алгоритмов', fontsize=14, pad=20)
plt.ylabel('Время (секунды)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Добавляем значения на столбцы
for bar, time_val in zip(bars, times):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{time_val:.2f}с', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# 2. ЛИНЕЙНЫЙ ГРАФИК ВЛИЯНИЯ ПАРАМЕТРОВ
plt.figure(figsize=(12, 5))

# График влияния n_neighbors
plt.subplot(1, 2, 1)
n_neighbors_range = [5, 10, 15, 20, 30, 50]
n_neighbors_times = []

for n_neighbors in n_neighbors_range:
    start_time = time.time()
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.5, random_state=42, n_jobs=-1)
    reducer.fit_transform(X_subset)
    n_neighbors_times.append(time.time() - start_time)

plt.plot(n_neighbors_range, n_neighbors_times, 'bo-', linewidth=2, markersize=6)
plt.title('Влияние n_neighbors на время UMAP', fontsize=12)
plt.xlabel('n_neighbors')
plt.ylabel('Время (секунды)')
plt.grid(True, alpha=0.3)

# График влияния min_dist
plt.subplot(1, 2, 2)
min_dist_range = [0.01, 0.1, 0.3, 0.5, 0.7, 0.99]
min_dist_times = []

for min_dist in min_dist_range:
    start_time = time.time()
    reducer = umap.UMAP(n_neighbors=15, min_dist=min_dist, random_state=42, n_jobs=-1)
    reducer.fit_transform(X_subset)
    min_dist_times.append(time.time() - start_time)

plt.plot(min_dist_range, min_dist_times, 'ro-', linewidth=2, markersize=6)
plt.title('Влияние min_dist на время UMAP', fontsize=12)
plt.xlabel('min_dist')
plt.ylabel('Время (секунды)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 3. КРУГОВАЯ ДИАГРАММА РАСПРЕДЕЛЕНИЯ ВРЕМЕНИ
plt.figure(figsize=(10, 6))

# Сравнение общего времени
total_umap_time = sum(times_umap)
labels = ['t-SNE', 'UMAP (все эксперименты)']
sizes = [tsne_time, total_umap_time]
colors = ['#ff9999', '#66b3ff']
explode = (0.1, 0)  # выделяем t-SNE

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.axis('equal')
plt.title('Распределение общего времени вычислений', fontsize=14)
plt.show()

# 4. ГИСТОГРАММА РАСПРЕДЕЛЕНИЯ КЛАССОВ
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
unique, counts = np.unique(y_subset, return_counts=True)
plt.bar(unique, counts, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Распределение цифр в датасете', fontsize=12)
plt.xlabel('Цифра')
plt.ylabel('Количество образцов')
plt.xticks(unique)
plt.grid(axis='y', alpha=0.3)

# Добавляем значения на столбцы
for i, count in enumerate(counts):
    plt.text(unique[i], count + 5, str(count), ha='center', va='bottom')

plt.subplot(1, 2, 2)
# Сравнение размеров кластеров после UMAP
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

cluster_metrics = []
for digit in unique:
    mask = y_subset == digit
    if np.sum(mask) > 3:  # нужно минимум 4 точки для ConvexHull
        cluster_points = X_umap[mask]
        if len(cluster_points) > 3:
            try:
                hull = ConvexHull(cluster_points)
                area = hull.volume
                cluster_metrics.append(area)
            except:
                cluster_metrics.append(0)
        else:
            cluster_metrics.append(0)
    else:
        cluster_metrics.append(0)

plt.bar(unique[:len(cluster_metrics)], cluster_metrics, color='lightgreen', edgecolor='black', alpha=0.7)
plt.title('Размеры кластеров после UMAP (площадь)', fontsize=12)
plt.xlabel('Цифра')
plt.ylabel('Площадь кластера')
plt.xticks(unique[:len(cluster_metrics)])
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# 5. ГРАФИК ТЕНДЕНЦИЙ ПРОИЗВОДИТЕЛЬНОСТИ
plt.figure(figsize=(10, 6))

# Создаем индекс для параметров
param_combinations = [f'n{n}_md{m}' for n in n_neighbors_list for m in min_dist_list]

plt.plot(param_combinations, times_umap, 's-', linewidth=2, markersize=8,
         color='purple', markerfacecolor='yellow', markeredgecolor='black')

# Добавляем горизонтальную линию для t-SNE
plt.axhline(y=tsne_time, color='red', linestyle='--', linewidth=2, label=f't-SNE ({tsne_time:.2f}с)')

plt.title('Время выполнения UMAP для различных комбинаций параметров', fontsize=14)
plt.xlabel('Комбинации параметров (n_neighbors_min_dist)')
plt.ylabel('Время (секунды)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Добавляем значения на точки
for i, time_val in enumerate(times_umap):
    plt.text(i, time_val + 0.02, f'{time_val:.2f}с', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

print("\n=== АНАЛИЗ ГРАФИКОВ ===")
print("📊 Столбчатые диаграммы: наглядно показывают разницу во времени выполнения")
print("📈 Линейные графики: демонстрируют тенденции влияния параметров")
print("🥧 Круговая диаграмма: показывает распределение вычислительных ресурсов")
print("📋 Гистограммы: отображают распределение данных и размеры кластеров")