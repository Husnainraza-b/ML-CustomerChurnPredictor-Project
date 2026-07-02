
from typing import List, Optional

import numpy as np
import pandas as pd


def one_hot_encode_categorical(
    df: pd.DataFrame,
    categorical_columns: Optional[List[str]] = None,
    drop_first: bool = True,
) -> pd.DataFrame:
   
    df = df.copy()

    if categorical_columns is None:
        categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

    categorical_columns = [c for c in categorical_columns if c in df.columns]

    if not categorical_columns:
        return df

    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=drop_first)
    return df_encoded


def min_max_scale_numeric(
    df: pd.DataFrame,
    numeric_columns: Optional[List[str]] = None,
    exclude_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
   
    df = df.copy()

    if exclude_columns is None:
        exclude_columns = ["CustomerID", "Churn"]

    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_columns = [c for c in numeric_columns if c not in exclude_columns]
    else:
        numeric_columns = [c for c in numeric_columns if c in df.columns and c not in exclude_columns]

    for col in numeric_columns:
        col_min = df[col].min()
        col_max = df[col].max()
        denom = col_max - col_min

        if denom == 0:
            df[col] = 0.0
        else:
            df[col] = (df[col] - col_min) / denom

    return df


if __name__ == "__main__":
    # Quick manual check when running this file directly.
    sample = pd.DataFrame({
        "CustomerID": [1, 2, 3],
        "Churn": [0, 1, 0],
        "Gender": ["Male", "Female", "Male"],
        "MaritalStatus": ["Single", "Married", "Divorced"],
        "Tenure": [5, 20, 12],
        "CashbackAmount": [100, 300, 200],
    })

    encoded = one_hot_encode_categorical(sample, ["Gender", "MaritalStatus"])
    print("After One-Hot Encoding:")
    print(encoded)

    scaled = min_max_scale_numeric(encoded)
    print("\nAfter Min-Max Scaling:")
    print(scaled)
