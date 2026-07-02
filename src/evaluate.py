import os
import sys
import pandas as pd

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.logistic_regression import LogisticRegression
from src.models.knn import KNearestNeighbors
from src.evaluation.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_models(filepath, lr_weights_path, knn_path, name="Dataset"):
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
    
    # Load Logistic Regression
    lr = LogisticRegression()
    lr.load_model(lr_weights_path)
    lr_pred = lr.predict(X)
    
    lr_acc = accuracy_score(y, lr_pred)
    lr_prec = precision_score(y, lr_pred)
    lr_rec = recall_score(y, lr_pred)
    lr_f1 = f1_score(y, lr_pred)
    
    # Load KNN
    knn = KNearestNeighbors()
    knn.load_model(knn_path)
    knn_pred = knn.predict(X)
    
    knn_acc = accuracy_score(y, knn_pred)
    knn_prec = precision_score(y, knn_pred)
    knn_rec = recall_score(y, knn_pred)
    knn_f1 = f1_score(y, knn_pred)
    
    print(f"\n=======================================================")
    print(f" Comparison Results for: {name}")
    print(f" File Path: {filepath}")
    print(f"=======================================================")
    print(f" Metric          | Logistic Regression | KNN (k={knn.k})")
    print(f" ----------------|---------------------|----------------")
    print(f" Accuracy        | {lr_acc:.4f}              | {knn_acc:.4f}")
    print(f" Precision       | {lr_prec:.4f}              | {knn_prec:.4f}")
    print(f" Recall          | {lr_rec:.4f}              | {knn_rec:.4f}")
    print(f" F1-Score        | {lr_f1:.4f}              | {knn_f1:.4f}")
    print(f"=======================================================\n")
    return {
        "lr": (lr_acc, lr_prec, lr_rec, lr_f1),
        "knn": (knn_acc, knn_prec, knn_rec, knn_f1)
    }

def main():
    processed_dir = os.path.join(project_root, "data", "processed")
    lr_weights_path = os.path.join(project_root, "src", "models", "logistic_regression_weights.npy")
    knn_path = os.path.join(project_root, "src", "models", "knn_model.npz")
    
    if not os.path.exists(lr_weights_path) or not os.path.exists(knn_path):
        print("Error: Trained model files not found.")
        print("Please train the models first by running: python src/train.py")
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
            evaluate_models(path, lr_weights_path, knn_path, name)
        else:
            print(f"Skipping {name}: file not found at {path}")


if __name__ == "__main__":
    main()
