import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Загрузка обработанных данных
df = pd.read_csv('processed_steam_data.csv')

# Разделение на признаки и целевую переменную
X = df.drop('popular', axis=1)
y = df['popular']

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Стратификация для сохранения баланса классов
)

print("РАЗДЕЛЕНИЕ ДАННЫХ НА ВЫБОРКИ")
print(f"Размер исходных данных: {X.shape}")
print(f"Размер тренировочной выборки: {X_train.shape} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Размер тестовой выборки: {X_test.shape} ({len(X_test)/len(X)*100:.1f}%)")

print(f"\nБаланс классов в тренировочной выборке:")
print(y_train.value_counts(normalize=True))

print(f"\nБаланс классов в тестовой выборке:")
print(y_test.value_counts(normalize=True))

# Сохранение разделенных данных
train_data = pd.DataFrame(X_train)
train_data['popular'] = y_train.values
test_data = pd.DataFrame(X_test)
test_data['popular'] = y_test.values

train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

print("\nДанные успешно разделены и сохранены!")
print("Файлы 'train_data.csv' и 'test_data.csv' созданы")