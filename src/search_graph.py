import pandas as pd
import networkx as nx


def build_search_graph(df: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()

    for _, row in df.iterrows():
        customer = row["customer"]
        merchant = row["merchant"]
        cost = row["cost"]
        risk = row["risk_score_scaled"]
        fraud = row["fraud"]
        amount = row["amount"]
        category = row["category"]

        G.add_edge(
            customer,
            merchant,
            cost=cost,
            risk=risk,
            fraud=fraud,
            amount=amount,
            category=category
        )

        G.add_edge(
            merchant,
            customer,
            cost=cost,
            risk=risk,
            fraud=fraud,
            amount=amount,
            category=category
        )

    return G


if __name__ == "__main__":
    input_file = "data/processed/banksim_risk_scored.csv"
    df = pd.read_csv(input_file)

    G = build_search_graph(df)

    print("Search graph built successfully.")
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    edge = list(G.edges(data=True))[0]
    print("\nExample edge:")
    print(edge)