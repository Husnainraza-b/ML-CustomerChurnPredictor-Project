import os
import sys
import pandas as pd

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.logistic_regression import LogisticRegression
from src.models.knn import KNearestNeighbors

def main():
    processed_dir = os.path.join(project_root, "data", "processed")
    train_path = os.path.join(processed_dir, "train_data.csv")
    
    if not os.path.exists(train_path):
        print(f"Error: Training file not found at {train_path}. Please run create_splits.py first.")
        return

    print("Loading training data...")
    train_df = pd.read_csv(train_path)
    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"]

    # Train Logistic Regression
    print("Training Logistic Regression model...")
    model = LogisticRegression(learning_rate=0.1, iterations=2000, verbose=True)
    model.fit(X_train, y_train)

    # Save model weights
    models_dir = os.path.join(project_root, "src", "models")
    weights_path = os.path.join(models_dir, "logistic_regression_weights.npy")
    model.save_model(weights_path)
    print(f"Logistic Regression weights saved to: {weights_path}")

    # Train K-Nearest Neighbors
    print("\nTraining K-Nearest Neighbors model...")
    knn_model = KNearestNeighbors(k=5)
    knn_model.fit(X_train, y_train)

    # Save KNN model parameters
    knn_path = os.path.join(models_dir, "knn_model.npz")
    knn_model.save_model(knn_path)
    print(f"KNN model parameters saved to: {knn_path}")


if __name__ == "__main__":
    main()
