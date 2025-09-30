import plotly.graph_objs as go
import pandas as pd
import numpy as np

# Загрузка данных
bmw_data = pd.read_csv('dataset/bmw_dataset.csv', delimiter=',')

print("=== ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ===")

# Группируем данные по модели и считаем среднюю цену
model_price_data = bmw_data.groupby('Model')['Price_USD'].mean().sort_values(ascending=False)

print("Данные для диаграммы (модели по средней цене):")
print(model_price_data.head(10))

print("\n=== ПОСТРОЕНИЕ КРУГОВОЙ ДИАГРАММЫ ===")

# Подготовка данных для круговой диаграммы
# Берем топ-5 моделей, а остальные объединяем в "Другие модели"
top_models = model_price_data.head(5)
other_models_sum = model_price_data[5:].sum()

# Создаем данные для круговой диаграммы
pie_labels = list(top_models.index) + ['Другие модели']
pie_values = list(top_models.values) + [other_models_sum]

# Создаем круговую диаграмму
fig = go.Figure()

fig.add_trace(go.Pie(
    labels=pie_labels,
    values=pie_values,
    hole=0.3,  # Делаем кольцевую диаграмму для лучшего вида
    marker=dict(
        line=dict(
            color='black', # Черные границы каждой доли
            width=2 # Толщина границы равна 2
        )
    ),
    textinfo='percent+label', # Показывать проценты и названия
    insidetextorientation='radial', # Читаемое расположение текста внутри
    textfont=dict(size=14) # Размер текста
))

# Настройка оформления
fig.update_layout(
    title=dict(
        text='Распределение средних цен по моделям BMW',
        x=0.5, # Заголовок по центру
        font=dict(size=20) # Размер текста 20
    ),
    height=700, # Высота 700 пикселей
    showlegend=True, # Показывать легенду для лучшей читаемости
    legend=dict(
        font=dict(size=14) # Размер текста в легенде
    ),
    margin=dict(l=50, r=50, t=100, b=50) # Убираем лишние отступы
)

# Показываем диаграмму
fig.show()

print("Круговая диаграмма успешно построена!")
print(f"Топ-5 моделей: {list(top_models.index)}")
print(f"Объединено в 'Другие модели': {len(model_price_data) - 5} моделей")
print(f"Общая сумма цен по другим моделям: ${other_models_sum:,.0f}")