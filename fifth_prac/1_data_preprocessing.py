import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Загрузка данных
df = pd.read_csv('dataset/steam_spy_data.csv')

# Просмотр информации о данных
print("Информация о датасете:")
print(df.info())
print("\nПервые 5 строк:")
print(df.head())

# Создание целевой переменной для бинарной классификации
# Будем классифицировать игры на "популярные" и "непопулярные"
# Используем медианное количество владельцев как порог
median_owners = df['owners'].str.extract('(\d+)').astype(float).median()
df['popular'] = (df['owners'].str.extract('(\d+)').astype(float) > median_owners).astype(int)

# Обработка категориальных признаков
le = LabelEncoder()
df['developer_encoded'] = le.fit_transform(df['developer'].fillna('Unknown'))
df['publisher_encoded'] = le.fit_transform(df['publisher'].fillna('Unknown'))

# Выбор признаков для модели
features = ['positive', 'negative', 'average_forever', 'average_2weeks',
           'median_forever', 'median_2weeks', 'price', 'discount', 'ccu',
           'developer_encoded', 'publisher_encoded']

X = df[features].fillna(0)  # Заполнение пропущенных значений
y = df['popular']

# Нормализация числовых признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nРазмерность признаков: {X_scaled.shape}")
print(f"Баланс классов: {y.value_counts()}")
print(f"Признаки: {features}")

# Сохранение обработанных данных
processed_data = pd.DataFrame(X_scaled, columns=features)
processed_data['popular'] = y.values
processed_data.to_csv('processed_steam_data.csv', index=False)

print("\nПредобработка данных завершена!")