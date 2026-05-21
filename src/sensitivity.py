
from __future__ import annotations

import copy
import random
import time
from collections import deque
from typing import Optional

import heapq
import numpy as np
import networkx as nx
import pandas as pd


def _bfs(G: nx.DiGraph, start, goal) -> dict:
    queue = deque([(start, [start])])
    visited = {start}
    nodes_expanded = 0
    t0 = time.time()

    while queue:
        current, path = queue.popleft()
        nodes_expanded += 1
        if current == goal:
            return {"path": path, "nodes_expanded": nodes_expanded,
                    "runtime": time.time() - t0}
        for nb in G.neighbors(current):
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, path + [nb]))

    return {"path": None, "nodes_expanded": nodes_expanded,
            "runtime": time.time() - t0}


def _astar(G: nx.DiGraph, start, goal) -> dict:
    pq = [(0, start, [start], 0)]
    best_cost: dict = {start: 0}
    nodes_expanded = 0
    t0 = time.time()

    while pq:
        f, current, path, g = heapq.heappop(pq)
        nodes_expanded += 1
        if current == goal:
            return {"path": path, "total_cost": g,
                    "nodes_expanded": nodes_expanded,
                    "runtime": time.time() - t0}
        for nb in G.neighbors(current):
            edge_cost = G[current][nb].get("cost", 1.0)
            new_cost = g + edge_cost
            if nb not in best_cost or new_cost < best_cost[nb]:
                best_cost[nb] = new_cost
                h = 1 - max(
                    (G[nb][n].get("risk", 0) for n in G.neighbors(nb)),
                    default=0,
                )
                heapq.heappush(pq, (new_cost + h, nb, path + [nb], new_cost))

    return {"path": None, "total_cost": None,
            "nodes_expanded": nodes_expanded, "runtime": time.time() - t0}


def _path_risk(G: nx.DiGraph, path: Optional[list]) -> float:
    if not path or len(path) < 2:
        return 0.0
    return sum(G[path[i]][path[i + 1]].get("risk", 0.0)
               for i in range(len(path) - 1))


def _path_len(path: Optional[list]) -> Optional[int]:
    return len(path) if path else None


def perturb_edge_weights(
    G: nx.DiGraph,
    noise_std: float = 0.05,
    field: str = "cost",
    seed: Optional[int] = None,
) -> nx.DiGraph:
    """
    Return a *copy* of G with Gaussian noise added to each edge's `field`.
    Values are clipped to [epsilon, 2.0] to keep costs positive.
    """
    rng = np.random.default_rng(seed)
    H = copy.deepcopy(G)
    for u, v, data in H.edges(data=True):
        original = data.get(field, 1.0)
        noise = rng.normal(0, noise_std)
        data[field] = float(np.clip(original + noise, 1e-4, 2.0))
    return H


def edge_perturbation_experiment(
    G: nx.DiGraph,
    start,
    goal,
    noise_levels: list[float] | None = None,
    n_trials: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    For each noise level, run n_trials perturbed graphs and collect:
      - algorithm, noise_std, trial
      - nodes_expanded, path_length, total_risk, runtime
    Returns a tidy DataFrame.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]

    records = []
    rng = random.Random(seed)

    for noise in noise_levels:
        for trial in range(n_trials):
            trial_seed = rng.randint(0, 10_000)
            H = perturb_edge_weights(G, noise_std=noise, seed=trial_seed)

            bfs_r = _bfs(H, start, goal)
            astar_r = _astar(H, start, goal)

            for algo, res in [("BFS", bfs_r), ("A*", astar_r)]:
                records.append({
                    "noise_std": noise,
                    "trial": trial,
                    "algorithm": algo,
                    "nodes_expanded": res["nodes_expanded"],
                    "path_length": _path_len(res["path"]),
                    "total_risk": _path_risk(H, res["path"]),
                    "runtime": res["runtime"],
                    "path_found": res["path"] is not None,
                    "total_cost": res.get("total_cost"),
                })

    return pd.DataFrame(records)




def remove_high_risk_nodes(
    G: nx.DiGraph,
    risk_threshold: float,
    start,
    goal,
) -> nx.DiGraph:

    H = copy.deepcopy(G)
    to_remove = []
    for node in list(H.nodes()):
        if node in (start, goal):
            continue
        max_risk = max(
            (H[node][nb].get("risk", 0) for nb in H.neighbors(node)),
            default=0,
        )
        if max_risk > risk_threshold:
            to_remove.append(node)
    H.remove_nodes_from(to_remove)
    return H, len(to_remove)


def node_removal_experiment(
    G: nx.DiGraph,
    start,
    goal,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:

    if thresholds is None:
        thresholds = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    records = []
    for thresh in thresholds:
        H, n_removed = remove_high_risk_nodes(G, thresh, start, goal)

        bfs_r = _bfs(H, start, goal)
        astar_r = _astar(H, start, goal)

        for algo, res in [("BFS", bfs_r), ("A*", astar_r)]:
            records.append({
                "risk_threshold": thresh,
                "nodes_removed": n_removed,
                "algorithm": algo,
                "nodes_expanded": res["nodes_expanded"],
                "path_length": _path_len(res["path"]),
                "total_risk": _path_risk(H, res["path"]),
                "runtime": res["runtime"],
                "path_found": res["path"] is not None,
            })

    return pd.DataFrame(records)




def risk_threshold_sweep(
    df: pd.DataFrame,
    start,
    goal,
    thresholds: list[float] | None = None,
    start_col: str = "customer",
    goal_col: str = "merchant",
    epsilon: float = 0.001,
) -> pd.DataFrame:
    
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.95, 18).tolist()

    records = []
    for thresh in thresholds:
        tmp = df.copy()
        tmp["flagged"] = (tmp["risk_score_scaled"] >= thresh).astype(int)
        tmp["adjusted_cost"] = np.where(
            tmp["flagged"] == 1,
            epsilon,                         
            epsilon + (1 - tmp["risk_score_scaled"]), 
        )

        H = nx.DiGraph()
        for _, row in tmp.iterrows():
            c, m = row[start_col], row[goal_col]
            kw = dict(
                cost=row["adjusted_cost"],
                risk=row["risk_score_scaled"],
                fraud=row["fraud"],
                amount=row["amount"],
                flagged=row["flagged"],
            )
            H.add_edge(c, m, **kw)
            H.add_edge(m, c, **kw)

        if start not in H or goal not in H:
            continue

        bfs_r = _bfs(H, start, goal)
        astar_r = _astar(H, start, goal)

        flagged_count = int(tmp["flagged"].sum())
        for algo, res in [("BFS", bfs_r), ("A*", astar_r)]:
            records.append({
                "threshold": round(thresh, 4),
                "flagged_edges": flagged_count,
                "algorithm": algo,
                "nodes_expanded": res["nodes_expanded"],
                "path_length": _path_len(res["path"]),
                "total_risk": _path_risk(H, res["path"]),
                "path_found": res["path"] is not None,
            })

    return pd.DataFrame(records)



def monte_carlo_perturbation(
    G: nx.DiGraph,
    start,
    goal,
    n_simulations: int = 100,
    noise_std: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    
    rng = random.Random(seed)
    records = []

    for i in range(n_simulations):
        trial_seed = rng.randint(0, 100_000)
        H = perturb_edge_weights(G, noise_std=noise_std, seed=trial_seed)

        bfs_r = _bfs(H, start, goal)
        astar_r = _astar(H, start, goal)

        for algo, res in [("BFS", bfs_r), ("A*", astar_r)]:
            records.append({
                "simulation": i,
                "algorithm": algo,
                "nodes_expanded": res["nodes_expanded"],
                "path_length": _path_len(res["path"]),
                "total_risk": _path_risk(H, res["path"]),
                "runtime": res["runtime"],
                "path_found": res["path"] is not None,
            })

    return pd.DataFrame(records)


def monte_carlo_summary(mc_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Monte Carlo results into mean ± std summary per algorithm."""
    numeric_cols = ["nodes_expanded", "path_length", "total_risk", "runtime"]
    rows = []
    for algo, grp in mc_df.groupby("algorithm"):
        row = {"algorithm": algo}
        for col in numeric_cols:
            row[f"{col}_mean"] = grp[col].mean()
            row[f"{col}_std"] = grp[col].std()
            row[f"{col}_min"] = grp[col].min()
            row[f"{col}_max"] = grp[col].max()
        row["path_found_rate"] = grp["path_found"].mean()
        rows.append(row)
    return pd.DataFrame(rows)



if __name__ == "__main__":
    df = pd.read_csv("data/processed/banksim_risk_scored.csv")

    from search_graph import build_search_graph
    G = build_search_graph(df)

    fraud_rows = df[df["fraud"] == 1]
    example = fraud_rows.iloc[0]
    start = example["customer"]
    goal = example["merchant"]

    print("=== Edge perturbation ===")
    ep_df = edge_perturbation_experiment(G, start, goal, n_trials=5)
    print(ep_df.groupby(["algorithm", "noise_std"])[["nodes_expanded", "total_risk"]].mean())

    print("\n=== Node removal ===")
    nr_df = node_removal_experiment(G, start, goal)
    print(nr_df[["risk_threshold", "nodes_removed", "algorithm", "nodes_expanded", "path_found"]])

    print("\n=== Monte Carlo ===")
    mc_df = monte_carlo_perturbation(G, start, goal, n_simulations=20)
    print(monte_carlo_summary(mc_df))