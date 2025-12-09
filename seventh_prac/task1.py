import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Load the dataset
df = pd.read_csv('datadet/steam_spy_data.csv')

# Select relevant numerical features and target
features = ['negative', 'userscore', 'average_forever', 'average_2weeks', 'median_forever', 'median_2weeks', 'price', 'initialprice', 'discount', 'ccu']
target = 'positive'

# Drop rows with missing values in selected columns
df = df.dropna(subset=features + [target])

# Extract features and target
X = df[features]
y = df[target]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaled data and scaler
with open('data.pkl', 'wb') as f:
    pickle.dump((X_train_scaled, X_test_scaled, y_train, y_test), f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Data preparation completed. Shapes:")
print(f"X_train: {X_train_scaled.shape}, X_test: {X_test_scaled.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")