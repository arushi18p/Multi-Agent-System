import pandas as pd
import networkx as nx
from collections import deque
import heapq
import time


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


def bfs_path(G, start, goal):
    start_time = time.time()
    queue = deque([(start, [start])])
    visited = {start}
    nodes_expanded = 0

    while queue:
        current, path = queue.popleft()
        nodes_expanded += 1

        if current == goal:
            return {
                "path": path,
                "nodes_expanded": nodes_expanded,
                "runtime": time.time() - start_time
            }

        for neighbor in G.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return {
        "path": None,
        "nodes_expanded": nodes_expanded,
        "runtime": time.time() - start_time
    }


def astar_path(G, start, goal):
    start_time = time.time()
    pq = [(0, start, [start], 0)]
    best_cost = {start: 0}
    nodes_expanded = 0

    while pq:
        f, current, path, g = heapq.heappop(pq)
        nodes_expanded += 1

        if current == goal:
            return {
                "path": path,
                "total_cost": g,
                "nodes_expanded": nodes_expanded,
                "runtime": time.time() - start_time
            }

        for neighbor in G.neighbors(current):
            edge_cost = G[current][neighbor].get("cost", 1.0)
            new_cost = g + edge_cost

            if neighbor not in best_cost or new_cost < best_cost[neighbor]:
                best_cost[neighbor] = new_cost
                heuristic = 0
                heapq.heappush(pq, (new_cost + heuristic, neighbor, path + [neighbor], new_cost))

    return {
        "path": None,
        "total_cost": None,
        "nodes_expanded": nodes_expanded,
        "runtime": time.time() - start_time
    }


def path_total_risk(G, path):
    if not path or len(path) < 2:
        return 0

    total = 0
    for i in range(len(path) - 1):
        total += G[path[i]][path[i + 1]].get("risk", 0)
    return total


if __name__ == "__main__":
    df = pd.read_csv("data/processed/banksim_risk_scored.csv")
    G = build_search_graph(df)

    fraud_rows = df[df["fraud"] == 1].copy()

    if fraud_rows.empty:
        print("No fraud rows found in sampled dataset.")
    else:
        example = fraud_rows.iloc[0]

        start = example["customer"]
        goal = example["merchant"]

        print("Using fraud-linked example")
        print("Start:", start)
        print("Goal:", goal)
        print("Fraud transaction amount:", example["amount"])
        print("Fraud transaction category:", example["category"])

        bfs_result = bfs_path(G, start, goal)
        astar_result = astar_path(G, start, goal)

        print("\nBFS Result:")
        print(bfs_result)
        print("BFS total risk:", path_total_risk(G, bfs_result["path"]))

        print("\nA* Result:")
        print(astar_result)
        print("A* total risk:", path_total_risk(G, astar_result["path"]))