import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    category_mean = df.groupby("category")["amount"].transform("mean")
    category_std = df.groupby("category")["amount"].transform("std")

    df["amount_zscore"] = (df["amount"] - category_mean) / category_std
    df["amount_zscore"] = df["amount_zscore"].fillna(0)

    df["customer_tx_count"] = df.groupby("customer")["amount"].transform("count")

    df["merchant_tx_count"] = df.groupby("merchant")["amount"].transform("count")

    return df


if __name__ == "__main__":

    input_file = "data/processed/banksim_sample.csv"
    output_file = "data/processed/banksim_features.csv"

    df = pd.read_csv(input_file)

    df = add_features(df)

    df.to_csv(output_file, index=False)

    print("Feature engineering complete.")
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nFirst rows:")
    print(df.head())