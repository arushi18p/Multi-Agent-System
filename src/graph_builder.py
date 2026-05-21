import pandas as pd
import networkx as nx


def build_graph(df: pd.DataFrame):
    G = nx.MultiDiGraph()

    for idx, row in df.iterrows():
        customer = row["customer"]
        merchant = row["merchant"]

        G.add_edge(
            customer,
            merchant,
            key=idx,
            amount=row["amount"],
            category=row["category"],
            risk=row["risk_score"],
            cost=row["cost"],
            fraud=row["fraud"]
        )

    return G


if __name__ == "__main__":
    input_file = "data/processed/banksim_risk_scored.csv"
    df = pd.read_csv(input_file)

    G = build_graph(df)

    print("Graph built successfully.")
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    edge = list(G.edges(keys=True, data=True))[0]
    print("\nExample edge:")
    print(edge)