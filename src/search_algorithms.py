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
        risk = row["risk_score"]
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
    visited = set([start])
    nodes_expanded = 0

    while queue:
        current, path = queue.popleft()
        nodes_expanded += 1

        if current == goal:
            runtime = time.time() - start_time
            return {
                "path": path,
                "nodes_expanded": nodes_expanded,
                "runtime": runtime
            }

        for neighbor in G.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    runtime = time.time() - start_time
    return {
        "path": None,
        "nodes_expanded": nodes_expanded,
        "runtime": runtime
    }


def astar_path(G, start, goal):
    start_time = time.time()

    pq = []
    heapq.heappush(pq, (0, start, [start], 0))
    best_cost = {start: 0}
    nodes_expanded = 0

    while pq:
        f, current, path, g = heapq.heappop(pq)
        nodes_expanded += 1

        if current == goal:
            runtime = time.time() - start_time
            return {
                "path": path,
                "total_cost": g,
                "nodes_expanded": nodes_expanded,
                "runtime": runtime
            }

        for neighbor in G.neighbors(current):
            edge_data = G[current][neighbor]
            edge_cost = edge_data.get("cost", 1.0)

            new_cost = g + edge_cost

            if neighbor not in best_cost or new_cost < best_cost[neighbor]:
                best_cost[neighbor] = new_cost

                heuristic = 1 - max((G[neighbor][n].get("risk", 0) for n in G.neighbors(neighbor)),default=0)
                f_score = new_cost + heuristic

                heapq.heappush(pq, (f_score, neighbor, path + [neighbor], new_cost))

    runtime = time.time() - start_time
    return {
        "path": None,
        "total_cost": None,
        "nodes_expanded": nodes_expanded,
        "runtime": runtime
    }


def path_total_risk(G, path):
    if not path or len(path) < 2:
        return 0

    total_risk = 0
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        total_risk += G[u][v].get("risk", 0)

    return total_risk


if __name__ == "__main__":
    input_file = "data/processed/banksim_risk_scored.csv"
    df = pd.read_csv(input_file)

    G = build_search_graph(df)

    nodes = list(G.nodes())
    start = nodes[0]
    goal = nodes[100]

    print(f"Start node: {start}")
    print(f"Goal node: {goal}")

    bfs_result = bfs_path(G, start, goal)
    astar_result = astar_path(G, start, goal)

    print("\nBFS Result:")
    print(bfs_result)
    print("BFS total risk:", path_total_risk(G, bfs_result["path"]))

    print("\nA* Result:")
    print(astar_result)
    print("A* total risk:", path_total_risk(G, astar_result["path"]))