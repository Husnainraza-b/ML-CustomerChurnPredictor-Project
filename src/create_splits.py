import os
import sys
import numpy as np
import pandas as pd

# Add the project root folder to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

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

def main():
    processed_dir = os.path.join(project_root, "data", "processed")
    cleaned_path = os.path.join(processed_dir, "cleaned_churn_data.csv")
    
    if not os.path.exists(cleaned_path):
        print(f"Error: {cleaned_path} not found. Run preprocessing first.")
        return
        
    df = pd.read_csv(cleaned_path)
    print(f"Loaded preprocessed data: {df.shape}")
    
    # Under-sample to balance the dataset
    df_balanced = random_undersample(df, target_col="Churn")
    print(f"Balanced data shape: {df_balanced.shape}")
    
    # Split into 70% Train, 15% Val, 15% Test
    train_df, val_df, test_df = train_val_test_split(df_balanced, train_size=0.7, val_size=0.15, random_state=42)
    
    train_path = os.path.join(processed_dir, "train_data.csv")
    val_path = os.path.join(processed_dir, "val_data.csv")
    test_path = os.path.join(processed_dir, "test_data.csv")
    
    # Save datasets
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print("\nSuccessfully created and saved datasets:")
    print(f"- Train split: {train_path} {train_df.shape}")
    print(f"- Val split:   {val_path} {val_df.shape}")
    print(f"- Test split:  {test_path} {test_df.shape}")

if __name__ == "__main__":
    main()
