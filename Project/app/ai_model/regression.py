import joblib
import numpy as np
import pandas as pd
import os

class LinearRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (-2 / n_samples) * np.dot(X.T, (y - y_pred))
            db = (-2 / n_samples) * np.sum(y - y_pred)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


def train_model_from_csv():
    # Path to dataset.csv inside the current file's directory
    csv_path = os.path.join(os.path.dirname(__file__), 'training.csv')

    # Load and validate dataset
    df = pd.read_csv(csv_path)
    required_columns = ['type', 'month', 'color', 'price', 'sells']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Dataset must contain columns: {', '.join(required_columns)}")

    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df, columns=['type', 'month', 'color'], drop_first=True)

    # Split features and target
    X_df = df_encoded.drop('sells', axis=1)
    y = df_encoded['sells'].values

    # Normalize features
    X_mean = X_df.mean()
    X_std = X_df.std()
    X = ((X_df - X_mean) / X_std).values

    # Train model
    model = LinearRegression(lr=0.01, epochs=1000)
    model.fit(X, y)

    # Save model and preprocessing data
    model_path = os.path.join(os.path.dirname(__file__), 'custom_linear_model.pkl')
    joblib.dump({
        'weights': model.weights,
        'bias': model.bias,
        'mean': X_mean.values,
        'std': X_std.values,
        'columns': X_df.columns.tolist()
    }, model_path)

    print(f"✅ Model trained and saved at: {model_path}")

# Uncomment this to train immediately when the script runs
if __name__ == "__main__":
    train_model_from_csv()
