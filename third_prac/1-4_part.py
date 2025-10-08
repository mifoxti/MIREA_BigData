import pandas as pd
import matplotlib.pyplot as plt

# 1. Загрузка данных
df = pd.read_csv('dataset/insurance.csv')

# 2. Статистика по данным
print("Статистика по данным:")
print(df.describe())

# 3. Гистограммы для числовых показателей
numeric_cols = ['age', 'bmi', 'children', 'charges']
df[numeric_cols].hist(figsize=(12, 8))
plt.tight_layout()
plt.show()

# 4. Меры центральной тенденции и разброса для bmi и charges
for col in ['bmi', 'charges']:
    data = df[col]
    mean = data.mean()
    median = data.median()
    mode = data.mode()[0]
    std = data.std()
    var = data.var()

    print(f"\n{col.upper()}:")
    print(f"Среднее: {mean:.2f}")
    print(f"Медиана: {median:.2f}")
    print(f"Мода: {mode:.2f}")
    print(f"Стандартное отклонение: {std:.2f}")
    print(f"Дисперсия: {var:.2f}")

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, alpha=0.7, label='Распределение')
    plt.axvline(mean, color='red', linestyle='--', label=f'Среднее: {mean:.2f}')
    plt.axvline(median, color='green', linestyle='--', label=f'Медиана: {median:.2f}')
    plt.axvline(mode, color='orange', linestyle='--', label=f'Мода: {mode:.2f}')
    plt.xlabel(col)
    plt.ylabel('Частота')
    plt.legend()
    plt.title(f'Распределение {col} с мерами центральной тенденции')
    plt.show()

