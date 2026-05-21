import pandas as pd


def preprocess_data(input_path: str, output_path: str, sample_size: int = 20000) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    keep_cols = [
        "step",
        "customer",
        "age",
        "gender",
        "merchant",
        "category",
        "amount",
        "fraud",
    ]
    df = df[keep_cols].copy()

    str_cols = ["customer", "age", "gender", "merchant", "category"]
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip("'")

    df = df.dropna()

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["fraud"] = pd.to_numeric(df["fraud"], errors="coerce")

    df = df.dropna(subset=["amount", "fraud"])

    df = df.sample(n=sample_size, random_state=42)

    df = df.reset_index(drop=True)

    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    input_file = "data/raw/banksim.csv"
    output_file = "data/processed/banksim_sample.csv"

    df = preprocess_data(input_file, output_file)

    print("Preprocessing complete.")
    print("Processed shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())