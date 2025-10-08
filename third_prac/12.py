import pandas as pd

df = pd.read_csv('dataset/ECDCCases.csv')

# 12. Поиск и удаление дубликатов
duplicates_count = df.duplicated().sum()
print(f"\nКоличество дубликатов: {duplicates_count}")
df = df.drop_duplicates()
print(f"Данные после удаления дубликатов: {df.shape}")
