"""
import pandas as pd

import joblib
import os
from regression import LinearRegression  # Your custom class

# Step 1: Load Data
csv_path = 'dataset.csv'
df = pd.read_csv(csv_path)

# Step 2: One-hot encode categorical features
df_encoded = pd.get_dummies(df, columns=['type', 'month', 'color'], drop_first=True)

# Step 3: Split features and target
X = df_encoded.drop('sells', axis=1).values
y = df_encoded['sells'].values

# Step 4: Normalize features (optional, helps gradient descent)
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Step 5: Train the model
model = LinearRegression(lr=0.01, epochs=1000)
model.fit(X, y)

# Step 6: Save model weights and scaler info
model_path = os.path.join(os.path.dirname(__file__), 'custom_linear_model.pkl')
joblib.dump({
    'weights': model.weights,
    'bias': model.bias,
    'mean': X.mean(axis=0),
    'std': X.std(axis=0),
    'columns': df_encoded.drop('sells', axis=1).columns.tolist()
}, model_path)

print(f"Model trained and saved at: {model_path}")
"""