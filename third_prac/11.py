import pandas as pd

df = pd.read_csv('dataset/ECDCCases.csv')

# 11. Статистика по данным и анализ выбросов
print("\nСтатистика по данным:")
print(df.describe())

# Анализ дней с количеством смертей > 3000
high_deaths = df[df['deaths'] > 3000]
if not high_deaths.empty:
    print(f"\nСтраны с количеством смертей > 3000 в день:")
    death_stats = high_deaths.groupby('countriesAndTerritories').size()
    print(death_stats)
else:
    print("\nДней с количеством смертей > 3000 не найдено")
