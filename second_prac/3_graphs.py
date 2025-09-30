import plotly.graph_objs as go
import pandas as pd
import numpy as np

bmw_data = pd.read_csv('dataset/bmw_dataset.csv', delimiter=',')

print("=== ПОДГОТОВКА ДАННЫХ ДЛЯ ДИАГРАММЫ ===")
# Группируем данные по модели и считаем среднюю цену
model_price_data = bmw_data.groupby('Model')['Price_USD'].mean().sort_values(ascending=False).head(10)

print("Данные для диаграммы (Топ-10 моделей по средней цене):")
print(model_price_data)

print("\n=== ПОСТРОЕНИЕ СТОЛБЧАТОЙ ДИАГРАММЫ ===")
# Создаем столбчатую диаграмму
fig = go.Figure()
# 3.1-3.3: Добавляем столбцы с настройками цвета и границ
fig.add_trace(go.Bar(
    x=model_price_data.index, # Названия моделей по оси X
    y=model_price_data.values, # Средние цены по оси Y
    marker=dict(
        color=model_price_data.values, # Цвет зависит от значения (3.2)
        coloraxis="coloraxis", # Цветовая шкала (3.2)
        line=dict(
            color='black', # Черная граница (3.3)
            width=2 # Толщина 2 (3.3)
        )
    ),
    # Показывать значения на столбцах
    text=model_price_data.values,
    texttemplate='%{text:,.0f}',
    textposition='outside'
))

# 3.4: Заголовок диаграммы
fig.update_layout(
    title=dict(
        text='Топ-10 моделей BMW по средней цене', # 3.4
        x=0.5, # Размещение по центру (3.4)
        xanchor='center', # Якорь по центру (3.4)
        font=dict(size=20) # Размер текста 20 (3.4)
    )
)

# 3.5-3.6: Настройки осей
fig.update_xaxes(
    title_text='Модель BMW', # Подпись оси X (3.5)
    title_font=dict(size=16), # Размер текста подписи 16 (3.5)
    tickangle=315, # Угол меток 315 градусов (3.5)
    tickfont=dict(size=14) # Размер текста меток 14 (3.6)
)

fig.update_yaxes(
    title_text='Средняя цена (USD)', # Подпись оси Y (3.5)
    title_font=dict(size=16), # Размер текста подписи 16 (3.5)
    tickfont=dict(size=14), # Размер текста меток 14 (3.6)
    # Форматирование числовых меток на оси Y
    tickformat=',.0f'
)

# 3.7: Размеры графика
fig.update_layout(
    width=None,     # Во всю ширину рабочей области (3.7)
    height=700      # Высота 700 пикселей (3.7)
)

# 3.8: Настройка сетки
fig.update_layout(
    plot_bgcolor='white', # Белый фон для лучшей видимости сетки
    xaxis=dict(
        gridcolor='ivory', # Цвет сетки (3.8)
        gridwidth=2 # Толщина сетки (3.8)
    ),
    yaxis=dict(
        gridcolor='ivory', # Цвет сетки (3.8)
        gridwidth=2 # Толщина сетки (3.8)
    )
)

# 3.9: Убираем лишние отступы
fig.update_layout(
    margin=dict(l=50, r=50, t=80, b=50) # Минимальные отступы (3.9)
)

# Цветовая шкала (дополнительная настройка для coloraxis)
fig.update_layout(
    coloraxis=dict(
        colorscale='Viridis',  # Можно выбрать другую палитру: 'Plasma', 'Inferno', 'Blues' и т.д.
        colorbar=dict(
            title="Цена (USD)",
            title_font=dict(size=14),
            tickfont=dict(size=12)
        )
    )
)

# Показываем диаграмму
fig.show()

print("Диаграмма успешно построена!")