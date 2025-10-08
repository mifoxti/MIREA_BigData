import pandas as pd
import scipy.stats as stats

# Создание датафрейма
data = pd.DataFrame({
    'Женат': [89, 17, 11, 43, 22, 1],
    'Гражданский брак': [80, 22, 20, 35, 6, 4],
    'Не состоит в отношениях': [35, 44, 35, 6, 8, 22]
})
data.index = ['Полный рабочий день', 'Частичная занятость', 'Временно не работает',
              'На домохозяйстве', 'На пенсии', 'Учёба']

print("Таблица сопряженности:")
print(data)

# Проверка гипотезы о независимости с помощью критерия хи-квадрат
chi2_stat, p_value, dof, expected = stats.chi2_contingency(data)

print(f"\nРезультаты критерия хи-квадрат:")
print(f"Хи-квадрат статистика: {chi2_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Степени свободы: {dof}")
print(f"\nОжидаемые частоты:")
print(pd.DataFrame(expected, index=data.index, columns=data.columns).round(2))

# Уровень значимости
alpha = 0.05
print(f"\nУровень значимости: {alpha}")

if p_value > alpha:
    print("Нет оснований отвергнуть нулевую гипотезу")
    print("Семейное положение и занятость НЕ зависят друг от друга")
else:
    print("Отвергаем нулевую гипотезу")
    print("Семейное положение и занятость ЗАВИСЯТ друг от друга")