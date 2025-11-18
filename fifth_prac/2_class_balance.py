import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка обработанных данных
df = pd.read_csv('processed_steam_data.csv')

# Настройка стиля графиков
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Создание гистограммы баланса классов
plt.figure(figsize=(10, 6))

# Гистограмма
plt.subplot(1, 2, 1)
class_counts = df['popular'].value_counts()
plt.bar(['Непопулярные (0)', 'Популярные (1)'], class_counts.values,
        color=['lightcoral', 'lightgreen'], alpha=0.7)
plt.title('Баланс классов игр', fontsize=14, fontweight='bold')
plt.xlabel('Класс')
plt.ylabel('Количество игр')
plt.grid(axis='y', alpha=0.3)

# Добавление значений на столбцы
for i, count in enumerate(class_counts.values):
    plt.text(i, count + 5, str(count), ha='center', va='bottom', fontweight='bold')

# Круговая диаграмма
plt.subplot(1, 2, 2)
plt.pie(class_counts.values, labels=['Непопулярные', 'Популярные'],
        autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightgreen'])
plt.title('Распределение классов', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('class_balance.png', dpi=300, bbox_inches='tight')
plt.show()

# Статистика по балансу классов
total_games = len(df)
popular_games = class_counts[1]
unpopular_games = class_counts[0]

print("АНАЛИЗ БАЛАНСА КЛАССОВ:")
print(f"Всего игр в датасете: {total_games}")
print(f"Популярные игры: {popular_games} ({popular_games/total_games*100:.2f}%)")
print(f"Непопулярные игры: {unpopular_games} ({unpopular_games/total_games*100:.2f}%)")
print(f"Соотношение классов: {popular_games/unpopular_games:.2f}:1")

# Выводы о балансе
if 0.8 <= popular_games/unpopular_games <= 1.2:
    print("\nВЫВОД: Классы сбалансированы")
else:
    print("\nВЫВОД: Классы несбалансированы")