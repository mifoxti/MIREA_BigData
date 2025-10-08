import scipy.stats as stats
import numpy as np

# Наблюдаемые частоты выпадений кубика
observed_frequencies = np.array([97, 98, 109, 95, 97, 104])

# Ожидаемые частоты при равномерном распределении (600 бросков / 6 граней)
expected_frequency = 600 / 6
expected_frequencies = np.array([expected_frequency] * 6)

print("Наблюдаемые частоты:", observed_frequencies)
print("Ожидаемые частоты:", expected_frequencies)

# Проверка гипотезы о равномерном распределении с помощью критерия хи-квадрат
chi2_stat, p_value = stats.chisquare(f_obs=observed_frequencies, f_exp=expected_frequencies)

print(f"\nРезультаты критерия хи-квадрат:")
print(f"Хи-квадрат статистика: {chi2_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Уровень значимости
alpha = 0.05
print(f"\nУровень значимости: {alpha}")

if p_value > alpha:
    print("Нет оснований отвергнуть нулевую гипотезу")
    print("Распределение можно считать равномерным")
else:
    print("Отвергаем нулевую гипотезу")
    print("Распределение не является равномерным")