
import os

import pandas as pd

try:
    
    from encoders import one_hot_encode_categorical, min_max_scale_numeric
except ImportError:
    from src.preprocessing.encoders import one_hot_encode_categorical, min_max_scale_numeric


def load_raw_data(filepath: str) -> pd.DataFrame:
   
    df = pd.read_excel(filepath, sheet_name='E Comm')
    return df


def impute_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
   
    if strategy not in ("median", "mean"):
        raise ValueError("strategy must be either 'median' or 'mean'")

    df = df.copy()

    columns_to_impute = [
        "Tenure",
        "DaySinceLastOrder",
        "WarehouseToHome",
        "HourSpendOnApp",
        "OrderAmountHikeFromlastYear",
        "CouponUsed",
        "OrderCount",
    ]

    for col in columns_to_impute:
        if col not in df.columns:
            continue
        if df[col].isna().sum() == 0:
            continue

        fill_value = df[col].median() if strategy == "median" else df[col].mean()
        df[col] = df[col].fillna(fill_value)

    return df


if __name__ == "__main__":
   
    RAW_PATH = "data/raw/E Commerce Dataset.xlsx"
    OUTPUT_DIR = "data/processed"
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, "cleaned_churn_data.csv")

    print(f"Loading raw data from: {RAW_PATH}")
    df = load_raw_data(RAW_PATH)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns.")

    print("\nMissing values BEFORE imputation:")
    missing_before = df.isna().sum()
    print(missing_before[missing_before > 0])

    print("\nImputing missing values (median strategy)...")
    df_clean = impute_missing_values(df, strategy="median")

    print("Missing values AFTER imputation:")
    missing_after = df_clean.isna().sum()
    remaining = missing_after[missing_after > 0]
    print(remaining if not remaining.empty else "None - all missing values handled.")

    
    if "CustomerID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["CustomerID"])

    print("\nOne-Hot Encoding categorical columns (Gender, MaritalStatus, etc.)...")
    df_encoded = one_hot_encode_categorical(df_clean)
    print(f"Shape after encoding: {df_encoded.shape}")

    print("\nMin-Max scaling numeric columns to [0, 1]...")
    df_scaled = min_max_scale_numeric(df_encoded)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_scaled.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved cleaned, encoded, and scaled data to: {OUTPUT_PATH}")
    print(f"Final shape: {df_scaled.shape}")
