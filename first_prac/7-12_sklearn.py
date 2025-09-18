import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

california_housing = fetch_california_housing()
data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
data['MedHouseVal'] = california_housing.target

print("8. Информация о датасете:")
print(data.info())
print("\n" + "="*50 + "\n")

print("9. Пропущенные значения:")
print(data.isna().sum())
print("\n" + "="*50 + "\n")

print("10. Записи где MedAge > 50 и Population > 2500:")
filtered_data = data[(data['HouseAge'] > 50) & (data['Population'] < 2500)]
print(filtered_data)
print(f"Количество записей: {len(filtered_data)}")
print("\n" + "="*50 + "\n")

print("11. Максимальное и минимальное значения MedHouseVal:")
print(f"Максимальное значение: {data['MedHouseVal'].max():.2f}")
print(f"Минимальное значение: {data['MedHouseVal'].min():.2f}")
print("\n" + "="*50 + "\n")

print("12. Средние значения признаков:")
mean_values = data.apply(lambda x: x.mean() if np.issubdtype(x.dtype, np.number) else None)
for feature, mean_val in mean_values.items():
    if mean_val is not None:
        print(f"{feature}: {mean_val:.4f}")