import pandas as pd

def random_undersample(df, target_col="Churn"):
    """
    Very simple random undersampler to balance the dataset.
    It finds the majority class and drops random rows
    until it matches the minority class.
    """
    df = df.copy()
    
    # Check if target column is in the dataframe
    if target_col not in df.columns:
        print("Target column not found!")
        return df
        
    # Find counts of each class
    class_counts = df[target_col].value_counts()
    
    # If there is only one class, do nothing
    if len(class_counts) < 2:
        return df
        
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    
    max_count = class_counts[majority_class]
    min_count = class_counts[minority_class]
    
    # Separate the majority and minority rows
    df_majority = df[df[target_col] == majority_class]
    df_minority = df[df[target_col] == minority_class]
    
    # How many rows we need to keep
    print(f"Downsampling majority class {majority_class} to {min_count} rows to balance the data...")
    
    # Sample randomly from the majority class without replacement
    df_majority_undersampled = df_majority.sample(n=min_count, replace=False, random_state=42)
    
    # Combine the data back together
    df_balanced = pd.concat([df_majority_undersampled, df_minority], axis=0)
    
    # Shuffle the rows so they are not all ordered by class
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df_balanced

if __name__ == "__main__":
    # simple test
    sample_data = pd.DataFrame({
        "Feature1": [1, 2, 3, 4, 5, 6, 7],
        "Churn": [0, 0, 0, 0, 0, 1, 1]  # imbalanced
    })
    
    print("Before balancing:")
    print(sample_data["Churn"].value_counts())
    
    balanced = random_undersample(sample_data, target_col="Churn")
    
    print("\nAfter balancing:")
    print(balanced["Churn"].value_counts())
