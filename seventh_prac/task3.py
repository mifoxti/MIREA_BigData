import pickle
import time
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
with open('data.pkl', 'rb') as f:
    X_train_scaled, X_test_scaled, y_train, y_test = pickle.load(f)

# Train XGBoost (Boosting)
start_time = time.time()
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train_scaled, y_train)
training_time = time.time() - start_time

# Predict
y_pred_train = xgb_model.predict(X_train_scaled)
y_pred_test = xgb_model.predict(X_test_scaled)

# Evaluate
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"XGBoost Training Time: {training_time:.2f} seconds")
print(f"Train MSE: {train_mse:.2f}, R2: {train_r2:.2f}")
print(f"Test MSE: {test_mse:.2f}, R2: {test_r2:.2f}")

# Save model and results
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

with open('xgb_results.pkl', 'wb') as f:
    pickle.dump((training_time, train_mse, test_mse, train_r2, test_r2), f)