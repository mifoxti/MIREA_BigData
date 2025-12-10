import pickle

# Load results
with open('rf_results.pkl', 'rb') as f:
    rf_times, rf_r2s, rf_importances, rf_train_mse, rf_test_mse, rf_train_r2, rf_test_r2 = pickle.load(f)

with open('xgb_results.pkl', 'rb') as f:
    xgb_times, xgb_r2s, xgb_importances, xgb_train_mse, xgb_test_mse, xgb_train_r2, xgb_test_r2 = pickle.load(f)

# Use the time for n=100
rf_time = rf_times[-1]
xgb_time = xgb_times[-1]

print("Сравнение бэггинга (Random Forest) и бустинга (XGBoost):")
print(f"\nRandom Forest:")
print(f"  Время обучения: {rf_time:.2f} секунд")
print(f"  Train MSE: {rf_train_mse:.2f}, R2: {rf_train_r2:.2f}")
print(f"  Test MSE: {rf_test_mse:.2f}, R2: {rf_test_r2:.2f}")

print(f"\nXGBoost:")
print(f"  Время обучения: {xgb_time:.2f} секунд")
print(f"  Train MSE: {xgb_train_mse:.2f}, R2: {xgb_train_r2:.2f}")
print(f"  Test MSE: {xgb_test_mse:.2f}, R2: {xgb_test_r2:.2f}")

# Выводы
print("\nВыводы:")
if rf_test_r2 > xgb_test_r2:
    print("Random Forest показал лучшие результаты на тестовых данных.")
elif xgb_test_r2 > rf_test_r2:
    print("XGBoost показал лучшие результаты на тестовых данных.")
else:
    print("Обе модели показали схожие результаты.")

if rf_time < xgb_time:
    print("Random Forest обучался быстрее.")
else:
    print("XGBoost обучался быстрее.")
