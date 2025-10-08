import pandas as pd
import scipy.stats as stats
import numpy as np

# Загрузка данных
df = pd.read_csv('dataset/bmi.csv')

# Выборки по регионам
northwest_bmi = df[df['region'] == 'northwest']['bmi']
southwest_bmi = df[df['region'] == 'southwest']['bmi']

print("Размеры выборок:")
print(f"Northwest: {len(northwest_bmi)}")
print(f"Southwest: {len(southwest_bmi)}")

# Проверка на нормальность (Шапиро-Уилк)
print("\nПроверка на нормальность (Шапиро-Уилк):")
shapiro_nw = stats.shapiro(northwest_bmi)
shapiro_sw = stats.shapiro(southwest_bmi)

print(f"Northwest: W={shapiro_nw.statistic:.4f}, p-value={shapiro_nw.pvalue:.4f}")
print(f"Southwest: W={shapiro_sw.statistic:.4f}, p-value={shapiro_sw.pvalue:.4f}")

# Проверка на гомогенность дисперсий (Бартлетт)
print("\nПроверка на гомогенность дисперсий (Бартлетт):")
bartlett_test = stats.bartlett(northwest_bmi, southwest_bmi)
print(f"T={bartlett_test.statistic:.4f}, p-value={bartlett_test.pvalue:.4f}")

# T-тест Стьюдента
print("\nT-тест Стьюдента для сравнения средних:")
t_test = stats.ttest_ind(northwest_bmi, southwest_bmi, equal_var=True)
print(f"t-статистика={t_test.statistic:.4f}, p-value={t_test.pvalue:.4f}")

# Описательные статистики
print("\nОписательные статистики:")
print(f"Northwest: среднее={np.mean(northwest_bmi):.2f}, ст.отклонение={np.std(northwest_bmi, ddof=1):.2f}")
print(f"Southwest: среднее={np.mean(southwest_bmi):.2f}, ст.отклонение={np.std(southwest_bmi, ddof=1):.2f}")