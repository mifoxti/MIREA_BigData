import pickle
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
with open('data.pkl', 'rb') as f:
    X_train_scaled, X_test_scaled, y_train, y_test = pickle.load(f)

# Features list (from task1.py)
features = ['negative', 'userscore', 'average_forever', 'average_2weeks', 'median_forever', 'median_2weeks', 'price', 'initialprice', 'discount', 'ccu']

# Range of n_estimators
n_estimators_list = [1, 5, 10, 20, 30, 50, 75, 100]
training_times = []
r2_scores = []

# Train models with different n_estimators
for n in n_estimators_list:
    start_time = time.time()
    xgb_model = XGBRegressor(n_estimators=n, random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time

    y_pred_test = xgb_model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred_test)

    training_times.append(training_time)
    r2_scores.append(r2)

    print(f"n_estimators: {n}, Training Time: {training_time:.4f} seconds, Test R2: {r2:.4f}")

# Train final model with 100 estimators for feature importance
xgb_model_final = XGBRegressor(n_estimators=100, random_state=42)
xgb_model_final.fit(X_train_scaled, y_train)
feature_importances = xgb_model_final.feature_importances_

# Predict with final model
y_pred_train = xgb_model_final.predict(X_train_scaled)
y_pred_test = xgb_model_final.predict(X_test_scaled)

# Evaluate final model
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"\nFinal XGBoost (n=100):")
print(f"Train MSE: {train_mse:.2f}, R2: {train_r2:.2f}")
print(f"Test MSE: {test_mse:.2f}, R2: {test_r2:.2f}")

# Create a single figure with three subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Feature Importance
axes[0].bar(range(len(features)), feature_importances)
axes[0].set_xticks(range(len(features)))
axes[0].set_xticklabels(features, rotation=45)
axes[0].set_title('Важность признаков')
axes[0].set_ylabel('Важность')

# Plot 2: Accuracy (R2) vs Number of Trees
axes[1].plot(n_estimators_list, r2_scores, marker='o')
axes[1].set_title('Точность (R2) от количества деревьев')
axes[1].set_xlabel('Количество деревьев (n_estimators)')
axes[1].set_ylabel('R2')
axes[1].grid(True)

# Plot 3: Training Time vs Number of Trees
axes[2].plot(n_estimators_list, training_times, marker='o')
axes[2].set_title('Время обучения от количества деревьев')
axes[2].set_xlabel('Количество деревьев (n_estimators)')
axes[2].set_ylabel('Время обучения (секунды)')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('xgb_all_plots.png')

# Save model and results
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model_final, f)

with open('xgb_results.pkl', 'wb') as f:
    pickle.dump((training_times, r2_scores, feature_importances, train_mse, test_mse, train_r2, test_r2), f)
