import os
import sys
import pandas as pd

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.logistic_regression import LogisticRegression

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

    print("Training Logistic Regression model...")
    model = LogisticRegression(learning_rate=0.1, iterations=2000, verbose=True)
    model.fit(X_train, y_train)

    # Save model weights
    models_dir = os.path.join(project_root, "src", "models")
    weights_path = os.path.join(models_dir, "logistic_regression_weights.npy")
    model.save_model(weights_path)
    print(f"Model trained successfully. Weights saved to: {weights_path}")

if __name__ == "__main__":
    main()
