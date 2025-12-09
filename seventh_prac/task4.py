import pickle

# Load results
with open('rf_results.pkl', 'rb') as f:
    rf_time, rf_train_mse, rf_test_mse, rf_train_r2, rf_test_r2 = pickle.load(f)

with open('xgb_results.pkl', 'rb') as f:
    xgb_time, xgb_train_mse, xgb_test_mse, xgb_train_r2, xgb_test_r2 = pickle.load(f)

print("Comparison of Bagging (Random Forest) and Boosting (XGBoost):")
print(f"\nRandom Forest:")
print(f"  Training Time: {rf_time:.2f} seconds")
print(f"  Train MSE: {rf_train_mse:.2f}, R2: {rf_train_r2:.2f}")
print(f"  Test MSE: {rf_test_mse:.2f}, R2: {rf_test_r2:.2f}")

print(f"\nXGBoost:")
print(f"  Training Time: {xgb_time:.2f} seconds")
print(f"  Train MSE: {xgb_train_mse:.2f}, R2: {xgb_train_r2:.2f}")
print(f"  Test MSE: {xgb_test_mse:.2f}, R2: {xgb_test_r2:.2f}")

# Conclusions
print("\nConclusions:")
if rf_test_r2 > xgb_test_r2:
    print("Random Forest performed better on test data.")
elif xgb_test_r2 > rf_test_r2:
    print("XGBoost performed better on test data.")
else:
    print("Both models performed similarly.")

if rf_time < xgb_time:
    print("Random Forest was faster to train.")
else:
    print("XGBoost was faster to train.")