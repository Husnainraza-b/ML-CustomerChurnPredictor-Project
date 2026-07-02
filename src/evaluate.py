import os
import sys
import pandas as pd

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.logistic_regression import LogisticRegression
from src.evaluation.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_dataset(filepath, weights_path, name="Dataset"):
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return
    
    # Load dataset
    df = pd.read_csv(filepath)
    if "Churn" not in df.columns:
        print(f"Error: 'Churn' target column missing in {filepath}")
        return
        
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    
    # Load trained model
    model = LogisticRegression()
    model.load_model(weights_path)
    
    # Predict
    y_pred = model.predict(X)
    
    # Compute metrics
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    print(f"\n==========================================")
    print(f"Evaluation Results for: {name}")
    print(f"File Path: {filepath}")
    print(f"==========================================")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"==========================================\n")
    return acc, prec, rec, f1

def main():
    processed_dir = os.path.join(project_root, "data", "processed")
    weights_path = os.path.join(project_root, "src", "models", "logistic_regression_weights.npy")
    
    if not os.path.exists(weights_path):
        print(f"Error: Trained model weights not found at {weights_path}.")
        print("Please train the model first by running: python src/train.py")
        return
        
    # Standard datasets
    datasets = {
        "Training Set": os.path.join(processed_dir, "train_data.csv"),
        "Validation Set": os.path.join(processed_dir, "val_data.csv"),
        "Test Set": os.path.join(processed_dir, "test_data.csv"),
    }
    
    # Run evaluation on all available files
    for name, path in datasets.items():
        if os.path.exists(path):
            evaluate_dataset(path, weights_path, name)
        else:
            print(f"Skipping {name}: file not found at {path}")

if __name__ == "__main__":
    main()
