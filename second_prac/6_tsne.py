import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import numpy as np

print("=== ВИЗУАЛИЗАЦИЯ МНОГОМЕРНЫХ ДАННЫХ С ПОМОЩЬЮ t-SNE ===")

# Загружаем готовый набор данных MNIST (упрощенная версия - digits)
digits = load_digits()
X, y = digits.data, digits.target

print(f"Размер данных: {X.shape}")
print(f"Количество классов: {len(np.unique(y))}")

# Берем подвыборку для ускорения вычислений
n_samples = 1000
X_subset = X[:n_samples]
y_subset = y[:n_samples]

print(f"Используем подвыборку: {X_subset.shape}")

# Параметры перплексии для исследования
perplexities = [5, 30, 50]

# Создаем график
plt.figure(figsize=(15, 5))

for i, perplexity in enumerate(perplexities):
    print(f"Вычисление t-SNE с перплексией = {perplexity}...")

    # Выполняем t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X_subset)

    # Строим график
    plt.subplot(1, 3, i + 1)
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_subset, cmap='tab10', s=20)
    plt.title(f'Перплексия = {perplexity}', fontsize=14)
    plt.xlabel('t-SNE компонента 1')
    plt.ylabel('t-SNE компонента 2')
    plt.colorbar(scatter, label='Класс')

plt.tight_layout()
plt.show()

print("\n=== ВЫВОД ===")
print("t-SNE визуализация показывает:")
print("1. Разные значения перплексии влияют на группировку кластеров")
print("2. Слишком малая перплексия (<10) - много мелких кластеров")
print("3. Слишком большая перплексия (>50) - кластеры могут сливаться")
print("4. Оптимальная перплексия обычно между 30-50 для большинства данных")