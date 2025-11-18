import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Загрузка данных
X_test = pd.read_csv('test_data.csv').drop('popular', axis=1)
y_test = pd.read_csv('test_data.csv')['popular']

# Загрузка моделей
models = {}
model_names = ['logistic_regression', 'svm', 'knn']

for name in model_names:
    models[name] = joblib.load(f'{name}_model.pkl')

print("ОЦЕНКА КАЧЕСТВА МОДЕЛЕЙ")
print("=" * 50)

# Создание таблицы для сравнения метрик
results = []

# Визуализация матриц ошибок
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, (name, model) in enumerate(models.items()):
    # Предсказания
    y_pred = model.predict(X_test)

    # Расчет метрик
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Сохранение результатов
    results.append({
        'Model': name.replace('_', ' ').title(),
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)

    # Визуализация матрицы ошибок
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Непопулярные', 'Популярные'],
                yticklabels=['Непопулярные', 'Популярные'],
                ax=axes[i])
    axes[i].set_title(f'Матрица ошибок: {name.replace("_", " ").title()}')
    axes[i].set_xlabel('Предсказанный класс')
    axes[i].set_ylabel('Фактический класс')

    # Отчет классификации
    print(f"\n{name.replace('_', ' ').upper()}:")
    print(classification_report(y_test, y_pred,
                                target_names=['Непопулярные', 'Популярные']))

# Сравнительная таблица метрик
results_df = pd.DataFrame(results)
print("\n" + "=" * 50)
print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА МЕТРИК:")
print("=" * 50)
print(results_df.round(4))

# Визуализация сравнения метрик
plt.figure(figsize=(12, 8))
metrics_plot = results_df.set_index('Model').plot(kind='bar', figsize=(12, 8))
plt.title('Сравнение метрик качества моделей', fontsize=16, fontweight='bold')
plt.ylabel('Значение метрики')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('models_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Определение лучшей модели
best_f1_model = results_df.loc[results_df['F1-Score'].idxmax()]
print(f"\nЛУЧШАЯ МОДЕЛЬ ПО F1-MEРЕ: {best_f1_model['Model']}")
print(f"F1-Score: {best_f1_model['F1-Score']:.4f}")

# Выводы
print("\nВЫВОДЫ:")
print("1. Все три алгоритма показали хорошие результаты на задаче классификации игр")
print("2. Метрика F1-мера является наиболее сбалансированной оценкой качества")
print("3. Рекомендуется использовать модель с наивысшим F1-Score для продакшена")