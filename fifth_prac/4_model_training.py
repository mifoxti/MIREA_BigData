import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import joblib

# Загрузка разделенных данных
X_train = pd.read_csv('train_data.csv').drop('popular', axis=1)
y_train = pd.read_csv('train_data.csv')['popular']
X_test = pd.read_csv('test_data.csv').drop('popular', axis=1)
y_test = pd.read_csv('test_data.csv')['popular']

print("НАЧАЛО ОБУЧЕНИЯ МОДЕЛЕЙ...")

# 1. Логистическая регрессия
print("\n1. ОБУЧЕНИЕ ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ...")
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

print("Логистическая регрессия обучена!")

# 2. SVM (метод опорных векторов)
print("\n2. ОБУЧЕНИЕ SVM...")
svm_model = SVC(random_state=42, probability=True)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

print("SVM модель обучена!")

# 3. KNN (k-ближайших соседей)
print("\n3. ОБУЧЕНИЕ KNN...")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

print("KNN модель обучена!")

# Сохранение предсказаний и моделей
models = {
    'Logistic Regression': (log_reg, y_pred_log_reg),
    'SVM': (svm_model, y_pred_svm),
    'KNN': (knn_model, y_pred_knn)
}

# Сохранение моделей
for name, (model, _) in models.items():
    joblib.dump(model, f'{name.lower().replace(" ", "_")}_model.pkl')

# Создание матриц ошибок
confusion_matrices = {}
for name, (_, y_pred) in models.items():
    confusion_matrices[name] = confusion_matrix(y_test, y_pred)

# Сохранение матриц ошибок
for name, matrix in confusion_matrices.items():
    matrix_df = pd.DataFrame(matrix,
                           index=['Факт: 0', 'Факт: 1'],
                           columns=['Прогноз: 0', 'Прогноз: 1'])
    matrix_df.to_csv(f'{name.lower().replace(" ", "_")}_confusion_matrix.csv')
    print(f"\nМатрица ошибок для {name}:")
    print(matrix_df)

print("\nВсе модели обучены и сохранены!")