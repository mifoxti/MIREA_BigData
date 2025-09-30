import pandas as pd
import numpy as np

# Настройки для лучшего отображения
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)  # Ширина вывода
pd.set_option('display.max_colwidth', 20)  # Максимальная ширина столбца

# Загрузка данных
bmw_data = pd.read_csv('dataset/bmw_dataset.csv', delimiter=',')

# Сброс индекса
bmw_data.reset_index(drop=True, inplace=True)

print("=== ПЕРВИЧНЫЙ АНАЛИЗ ДАННЫХ BMW ===")

# 1. Первый взгляд на данные
print("\n1. Первые 5 строк набора данных:")
print(bmw_data.head())

print("\n2. Последние 3 строки (полезно для проверки целостности):")
print(bmw_data.tail(3))

# 2. Базовая информация
print(f"\n3. Размерность данных: {bmw_data.shape}")
print(f"   - Количество записей: {bmw_data.shape[0]}")
print(f"   - Количество признаков: {bmw_data.shape[1]}")

print("\n4. Информация о столбцах и типах данных:")
bmw_data.info()

# 3. Статистический анализ
print("\n5. Основные статистики для числовых столбцов:")
print(bmw_data.describe())

# Добавим статистики для строковых столбцов
print("\n6. Статистики для категориальных столбцов:")
categorical_columns = bmw_data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    print(f"\n   {col}:")
    print(f"   Уникальных значений: {bmw_data[col].nunique()}")
    print(f"   Топ-5 самых частых:")
    print(bmw_data[col].value_counts().head())

# 4. Анализ пропущенных значений
print("\n7. Анализ пропущенных значений:")
missing_data = bmw_data.isnull().sum()
missing_percent = (bmw_data.isnull().sum() / len(bmw_data)) * 100
missing_info = pd.DataFrame({
    'Количество пропусков': missing_data,
    'Процент пропусков': missing_percent.round(2)
})
print(missing_info[missing_info['Количество пропусков'] > 0])

if missing_info['Количество пропусков'].sum() == 0:
    print("   ✓ Пропущенных значений нет!")

# 5. Анализ распределения ключевых переменных
print("\n8. Распределение цен (Price_USD):")
if 'Price_USD' in bmw_data.columns:
    price_stats = bmw_data['Price_USD'].describe()
    print(f"   Диапазон цен: ${price_stats['min']:,.0f} - ${price_stats['max']:,.0f}")
    print(f"   Средняя цена: ${price_stats['mean']:,.0f}")
    print(f"   Медианная цена: ${price_stats['50%']:,.0f}")

    # Более детальное распределение
    print("\n   Детальное распределение цен:")
    print(bmw_data['Price_USD'].value_counts(bins=10).sort_index())

print("\n9. Распределение по моделям:")
if 'Model' in bmw_data.columns:
    model_counts = bmw_data['Model'].value_counts()
    print(f"   Всего уникальных моделей: {len(model_counts)}")
    print(f"   Топ-10 самых популярных моделей:")
    print(model_counts.head(10))

# 6. Дополнительные полезные проверки
print("\n10. Дополнительная информация:")
print(f"   Дубликатов в данных: {bmw_data.duplicated().sum()}")
print(f"   Потребление памяти: {bmw_data.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

# 7. Быстрый анализ корреляций
numeric_columns = bmw_data.select_dtypes(include=[np.number]).columns
if len(numeric_columns) > 1:
    print(f"\n11. Корреляция числовых признаков (топ-5 самых сильных):")
    correlations = bmw_data[numeric_columns].corr()
    # Уберем диагональные элементы (всегда 1)
    if 'Price_USD' in correlations.columns:
        price_correlations = correlations['Price_USD'].drop('Price_USD', errors='ignore')
        top_correlations = price_correlations.abs().sort_values(ascending=False).head(6)
        print("   Самые значимые корреляции с ценой:")
        for feature, corr in price_correlations.loc[top_correlations.index].items():
            print(f"   {feature}: {corr:.3f}")

print("\n=== АНАЛИЗ ЗАВЕРШЕН ===")