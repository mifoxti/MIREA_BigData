import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv('dataset/steam_spy_data.csv')

print("Информация о датасете:")
print(df.info())
print("\nПервые 5 строк:")
print(df.head())

median_owners = df['owners'].str.extract('(\d+)').astype(float).median()
df['popular'] = (df['owners'].str.extract('(\d+)').astype(float) > median_owners).astype(int)

le = LabelEncoder()
df['developer_encoded'] = le.fit_transform(df['developer'].fillna('Unknown'))
df['publisher_encoded'] = le.fit_transform(df['publisher'].fillna('Unknown'))

features = ['positive', 'negative', 'average_forever', 'average_2weeks',
           'median_forever', 'median_2weeks', 'price', 'discount', 'ccu',
           'developer_encoded', 'publisher_encoded']

X = df[features].fillna(0)
y = df['popular']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nРазмерность признаков: {X_scaled.shape}")
print(f"Баланс классов: {y.value_counts()}")
print(f"Признаки: {features}")

processed_data = pd.DataFrame(X_scaled, columns=features)
processed_data['popular'] = y.values
processed_data.to_csv('processed_steam_data.csv', index=False)

print("\nПредобработка данных завершена!")