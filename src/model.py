import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def train_model(df: pd.DataFrame):

    # Features used for prediction
    feature_cols = [
        "amount",
        "amount_zscore",
        "customer_tx_count",
        "merchant_tx_count"
    ]

    X = df[feature_cols]
    y = df["fraud"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    # Evaluate model
    predictions = model.predict(X_test)

    print("\nModel evaluation:")
    print(classification_report(y_test, predictions))

    return model


if __name__ == "__main__":

    input_file = "data/processed/banksim_features.csv"

    df = pd.read_csv(input_file)

    model = train_model(df)

    # Generate fraud probability for every transaction
    feature_cols = [
        "amount",
        "amount_zscore",
        "customer_tx_count",
        "merchant_tx_count"
    ]

    df["risk_score"] = model.predict_proba(df[feature_cols])[:, 1]

    max_risk = df["risk_score"].max()
    if max_risk > 0:
        df["risk_score_scaled"] = df["risk_score"] / max_risk
    else:
        df["risk_score_scaled"] = 0.0

    epsilon = 0.001
    df["cost"] = epsilon + (1 - df["risk_score_scaled"])

    output_file = "data/processed/banksim_risk_scored.csv"
    df.to_csv(output_file, index=False)

    print("\nRisk scoring complete.")
    print("Saved file:", output_file)

    print("\nPreview:")
    print(df[["amount", "risk_score", "risk_score_scaled", "cost"]].head())

    print("\nRisk summary:")
    print(df["risk_score"].describe())
    print("\nScaled risk summary:")
    print(df["risk_score_scaled"].describe())
    print("\nTop 10 scaled risks:")
    print(df["risk_score_scaled"].sort_values(ascending=False).head(10))