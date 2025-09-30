import pandas as pd
import numpy as np

# Настройки для лучшего отображения
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)  # Ширина вывода
pd.set_option('display.max_colwidth', 20)  # Максимальная ширина столбца

# Загрузка данных
bmw_data = pd.read_csv('dataset/bmw_dataset.csv', delimiter=',')

# Сброс индекса - хорошая практика после загрузки
bmw_data.reset_index(drop=True, inplace=True)

print("=== ПЕРВИЧНЫЙ АНАЛИЗ ДАННЫХ BMW ===")

print("\n1. .info() набора данных:")
print(bmw_data.info())

print("\n2. Первые 5 строк набора данных:")
print(bmw_data.head())

print("\n3. Анализ пропущенных значений:")
missing_data = bmw_data.isnull().sum()
missing_percent = (bmw_data.isnull().sum() / len(bmw_data)) * 100
missing_info = pd.DataFrame({
    'Количество пропусков': missing_data,
    'Процент пропусков': missing_percent.round(2)
})
print(missing_info[missing_info['Количество пропусков'] > 0])

if missing_info['Количество пропусков'].sum() == 0:
    print("   ✓ Пропущенных значений нет!")
