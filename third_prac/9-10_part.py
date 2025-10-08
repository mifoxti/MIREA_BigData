import pandas as pd

# 9. Загрузка данных
df = pd.read_csv('dataset/ECDCCases.csv')

# 10. Проверка и обработка пропущенных значений
print("Пропущенные значения в %:")
missing_percent = (df.isnull().sum() / len(df)) * 100
print(missing_percent.sort_values(ascending=False))

# Удаление двух признаков с наибольшим количеством пропусков
columns_to_drop = missing_percent.nlargest(2).index
df = df.drop(columns=columns_to_drop)
print(f"\nУдалены признаки: {list(columns_to_drop)}")

# Обработка оставшихся пропусков
for column in df.columns:
    if df[column].isnull().sum() > 0:
        if df[column].dtype == 'object':
            df[column] = df[column].fillna('other')
        else:
            df[column] = df[column].fillna(df[column].median())

print("\nПропуски после обработки:")
print(df.isnull().sum())
