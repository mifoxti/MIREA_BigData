import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Загрузка данных
bmw_data = pd.read_csv('dataset/bmw_dataset.csv', delimiter=',')

print("=== ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ===")

# Выбираем числовые параметры для анализа
print("Доступные числовые столбцы:")
print(bmw_data.select_dtypes(include=[np.number]).columns.tolist())

# Сортируем данные по основному параметру
main_param = 'Year'

# Выбираем 2-5 зависимых показателей
dependent_params = []
for col in ['Price_USD', 'Mileage_KM', 'Engine_Size_L', 'Price_USD', 'Sales_Volume']:
    if col in bmw_data.columns:
        dependent_params.append(col)
    if len(dependent_params) >= 4:  # Берем до 4 показателей
        break

print(f"\nОсновной параметр: {main_param}")
print(f"Зависимые показатели: {dependent_params}")

# Группируем данные по основному параметру и считаем средние значения
grouped_data = bmw_data.groupby(main_param)[dependent_params].mean().reset_index()

print("\nПодготовленные данные для графиков:")
print(grouped_data.head())

print("\n=== ПОСТРОЕНИЕ ЛИНЕЙНЫХ ГРАФИКОВ ===")

# Создаем график
plt.figure(figsize=(12, 8))

# Для каждого зависимого показателя строим линию
for i, param in enumerate(dependent_params):
    # 5.1: Линия с маркерами и заданными цветами
    plt.plot(
        grouped_data[main_param],
        grouped_data[param],
        marker='o', # Маркеры в виде кружков
        color='crimson', # Цвет линии
        markerfacecolor='white', # Цвет точек внутри
        markeredgecolor='black', # Цвет границ точек
        markeredgewidth=2, # Толщина границ точек равна 2
        linewidth=2, # Толщина линии
        label=param # Подпись для легенды
    )

# 5.2: Добавляем сетку
plt.grid(
    True,
    color='mistyrose', # Цвет сетки
    linewidth=2 # Толщина сетки
)

# Настройка оформления
plt.title(f'Зависимость показателей от параметра "{main_param}"', fontsize=16, pad=20)
plt.xlabel(main_param, fontsize=14)
plt.ylabel('Значения показателей', fontsize=14)
plt.legend(fontsize=12)

# Улучшаем читаемость
plt.tight_layout()

# Показываем график
plt.show()

print("\n=== ВЫВОД ===")
print("На графике показаны зависимости различных параметров автомобилей BMW от основного параметра.")
print("Анализ позволяет выявить:")
print("1. Тренды изменения показателей со временем/изменением основного параметра")
print("2. Взаимосвязи между различными характеристиками")
print("3. Аномальные значения или выбросы в данных")

# Дополнительная статистика для вывода
if not grouped_data.empty:
    print(f"\nСтатистика по основному параметру '{main_param}':")
    print(f"- Диапазон: от {grouped_data[main_param].min()} до {grouped_data[main_param].max()}")
    print(f"- Количество уникальных значений: {len(grouped_data)}")