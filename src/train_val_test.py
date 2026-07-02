import os
import sys
import numpy as np
import pandas as pd

# Add the project root folder to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.logistic_regression import LogisticRegression
from src.evaluation.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.preprocessing.imbalance import random_undersample

def train_val_test_split(df, train_size=0.7, val_size=0.15, random_state=42):
    """
    Splits the dataframe into training, validation, and testing sets from scratch using numpy.
    """
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(df))
    
    n_train = int(len(df) * train_size)
    n_val = int(len(df) * val_size)
    
    train_indices = shuffled_indices[:n_train]
    val_indices = shuffled_indices[n_train:n_train + n_val]
    test_indices = shuffled_indices[n_train + n_val:]
    
    return df.iloc[train_indices], df.iloc[val_indices], df.iloc[test_indices]

def evaluate_predictions(y_true, y_pred, dataset_name="Dataset"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\n--- {dataset_name} Evaluation Results ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    return acc, prec, rec, f1

def main():
    processed_dir = os.path.join(project_root, "data", "processed")
    
    train_path = os.path.join(processed_dir, "train_data.csv")
    val_path = os.path.join(processed_dir, "val_data.csv")
    test_path = os.path.join(processed_dir, "test_data.csv")
    cleaned_path = os.path.join(processed_dir, "cleaned_churn_data.csv")
    
    # Check if Abdullah's split files exist
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        print("Found custom training, validation, and test datasets. Loading them...")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
    else:
        print("Custom split files not found. Using cleaned_churn_data.csv and creating split...")
        if not os.path.exists(cleaned_path):
            print(f"Error: Processed dataset not found at {cleaned_path}.")
            print("Please run clean_data.py first.")
            return
            
        df = pd.read_csv(cleaned_path)
        print(f"Loaded dataset with shape: {df.shape}")
        
        # Balance dataset
        df = random_undersample(df, target_col="Churn")
        
        # Split: 70% Train, 15% Val, 15% Test
        train_df, val_df, test_df = train_val_test_split(df, train_size=0.7, val_size=0.15, random_state=42)
        
    print(f"Train set shape: {train_df.shape}")
    print(f"Val set shape:   {val_df.shape}")
    print(f"Test set shape:  {test_df.shape}")

    # Prepare features and target
    if "Churn" not in train_df.columns:
        print("Error: 'Churn' target column not found in dataset.")
        return
        
    X_train, y_train = train_df.drop(columns=["Churn"]), train_df["Churn"]
    X_val, y_val = val_df.drop(columns=["Churn"]), val_df["Churn"]
    X_test, y_test = test_df.drop(columns=["Churn"]), test_df["Churn"]

    # Train model
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(learning_rate=0.1, iterations=2000, verbose=False)
    model.fit(X_train, y_train)

    # Evaluate on Train
    y_train_pred = model.predict(X_train)
    evaluate_predictions(y_train, y_train_pred, "Training Set")

    # Evaluate on Val
    y_val_pred = model.predict(X_val)
    evaluate_predictions(y_val, y_val_pred, "Validation Set")

    # Evaluate on Test
    y_pred = model.predict(X_test)
    evaluate_predictions(y_test, y_pred, "Test Set")

if __name__ == "__main__":
    main()
