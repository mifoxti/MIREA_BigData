import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Загрузка набора данных
data = pd.read_csv('dataset/steam_spy_data.csv')

# Функция для парсинга диапазона владельцев и вычисления среднего
def parse_owners(owners_str):
    if '..' in owners_str:
        parts = owners_str.split('..')
        low = int(parts[0].replace(',', '').strip())
        high = int(parts[1].replace(',', '').strip())
        return (low + high) / 2
    else:
        return int(owners_str.replace(',', '').strip())

# Применение к столбцу владельцев
data['owners_avg'] = data['owners'].apply(parse_owners)

# Выбор числовых столбцов для кластеризации
numerical_cols = ['positive', 'negative', 'userscore', 'owners_avg', 'average_forever', 'average_2weeks',
                  'median_forever', 'median_2weeks', 'price', 'initialprice', 'discount', 'ccu']

# Удаление строк с пропущенными значениями в выбранных столбцах
data_clean = data[numerical_cols].dropna()

# Нормализация данных
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data_clean)

# Сохранение нормализованных данных для использования в других задачах
np.savetxt('prepared_data.csv', data_normalized, delimiter=',')
print("Подготовка данных и нормализация завершена. Сохранено в prepared_data.csv")