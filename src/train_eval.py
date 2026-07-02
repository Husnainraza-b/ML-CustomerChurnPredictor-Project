import os
import sys
import numpy as np
import pandas as pd

# Add the project root folder to sys.path so we can import src modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.logistic_regression import LogisticRegression
from src.evaluation.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.preprocessing.imbalance import random_undersample

def train_test_split(df, test_size=0.2, random_state=42):
    """
    Splits the dataframe into training and testing sets from scratch using numpy.
    """
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(df))
    test_set_size = int(len(df) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return df.iloc[train_indices], df.iloc[test_indices]

def main():
    processed_data_path = os.path.join(project_root, "data", "processed", "cleaned_churn_data.csv")
    
    if not os.path.exists(processed_data_path):
        print(f"Processed dataset not found at {processed_data_path}.")
        print("Please ensure the raw data is cleaned and saved first by running clean_data.py.")
        return

    print(f"Loading processed data from {processed_data_path}...")
    df = pd.read_csv(processed_data_path)
    print(f"Loaded dataset with shape: {df.shape}")

    # Balance the dataset (under-sample majority class)
    df = random_undersample(df, target_col="Churn")
    print(f"Dataset shape after balancing: {df.shape}")

    # Separate features and target
    if "Churn" not in df.columns:
        print("Error: 'Churn' target column not found in dataset.")
        return
        
    # Split into train and test sets (80% train, 20% test)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Training set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")

    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"]
    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]

    # Initialize and train model
    print("\nTraining Logistic Regression model from scratch using Gradient Descent...")
    model = LogisticRegression(learning_rate=0.1, iterations=2000, verbose=True)
    model.fit(X_train, y_train)

    # Make predictions
    print("\nEvaluating model on test set...")
    y_pred = model.predict(X_test)

    # Calculate metrics using our custom functions
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n==================================")
    print("--- Test Set Evaluation Results ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("==================================")

if __name__ == "__main__":
    main()
