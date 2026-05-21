import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from collections import deque
import heapq
import time
import matplotlib.pyplot as plt
import sys
import os
from agents.graph import build_graph

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from sensitivity import (
    edge_perturbation_experiment,
    node_removal_experiment,
    risk_threshold_sweep,
    monte_carlo_perturbation,
    monte_carlo_summary,
)


st.set_page_config(page_title="Fraud Network Analysis", layout="wide")


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
                heuristic = 1 - max((G[neighbor][n].get("risk", 0) for n in G.neighbors(neighbor)),default=0)
                heapq.heappush(
                    pq,
                    (new_cost + heuristic, neighbor, path + [neighbor], new_cost)
                )

    return {
        "path": None,
        "total_cost": None,
        "nodes_expanded": nodes_expanded,
        "runtime": time.time() - start_time
    }


def path_total_risk(G, path):
    if not path or len(path) < 2:
        return 0.0

    total = 0.0
    for i in range(len(path) - 1):
        total += G[path[i]][path[i + 1]].get("risk", 0.0)
    return total


def path_to_table(G, path):
    if not path or len(path) < 2:
        return pd.DataFrame()

    rows = []
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        edge = G[u][v]

        rows.append({
            "from": u,
            "to": v,
            "amount": edge.get("amount"),
            "category": edge.get("category"),
            "risk": edge.get("risk"),
            "fraud": edge.get("fraud")
        })

    return pd.DataFrame(rows)


def format_path(path):
    if not path:
        return "No path found"
    return " → ".join(path)


def make_comparison_chart(bfs_nodes, astar_nodes, bfs_runtime, astar_runtime):
    nodes_df = pd.DataFrame({
        "Algorithm": ["BFS", "A*"],
        "Nodes Expanded": [bfs_nodes, astar_nodes]
    })

    runtime_df = pd.DataFrame({
        "Algorithm": ["BFS", "A*"],
        "Runtime (s)": [bfs_runtime, astar_runtime]
    })

    fig1, ax1 = plt.subplots(figsize=(5, 4))
    ax1.bar(nodes_df["Algorithm"], nodes_df["Nodes Expanded"])
    ax1.set_title("Nodes Expanded")
    ax1.set_ylabel("Count")

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.bar(runtime_df["Algorithm"], runtime_df["Runtime (s)"])
    ax2.set_title("Runtime Comparison")
    ax2.set_ylabel("Seconds")

    return fig1, fig2


# -----------------------------
# Improvement Helpers
# -----------------------------
def node_label(node):
    if str(node).startswith("C"):
        return f"{node} (Customer)"
    if str(node).startswith("M"):
        return f"{node} (Merchant)"
    return str(node)


def style_fraud_column(df_table: pd.DataFrame):
    if df_table.empty or "fraud" not in df_table.columns:
        return df_table

    return df_table.style.map(
        lambda v: "background-color: #8b0000; color: white;" if v == 1 else "",
        subset=["fraud"]
    )



@st.cache_data
def load_data():
    return pd.read_csv("data/processed/banksim_risk_scored.csv")


df = load_data()
G = build_search_graph(df)
agent_graph = build_graph()
fraud_rows = df[df["fraud"] == 1].copy()



st.title("Fraud Network Analysis")

st.write(
    "This demo compares uninformed BFS with cost-guided A* on a BankSim-based transaction graph."
)

st.info(
    """
This system demonstrates how fraud detection signals can be integrated into a graph-based search framework.

- Transactions are modeled as edges between customers and merchants  
- A machine learning model assigns fraud-risk scores  
- BFS explores shortest structural paths  
- A* explores paths guided by fraud-risk cost
"""
)


st.sidebar.header("Controls")

mode = st.sidebar.selectbox(
    "Example type",
    ["Fraud-linked example", "Manual customer/merchant selection"]
)

if mode == "Fraud-linked example" and not fraud_rows.empty:
    fraud_index = st.sidebar.slider(
        "Fraud example index",
        min_value=0,
        max_value=len(fraud_rows) - 1,
        value=0
    )
    example = fraud_rows.iloc[fraud_index]
    start = example["customer"]
    goal = example["merchant"]
else:
    customer_options = sorted(df["customer"].unique().tolist())
    merchant_options = sorted(df["merchant"].unique().tolist())
    start = st.sidebar.selectbox("Start customer", customer_options)
    goal = st.sidebar.selectbox("Goal merchant", merchant_options)

run_search = st.sidebar.button("Run Analysis")


tab_main, tab_sensitivity, tab_agents = st.tabs(["Search Analysis", "Sensitivity Analysis", "Multi-Agent Workflow"])


with tab_main:

    st.subheader("Dataset Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", len(df))
    c2.metric("Customers", df["customer"].nunique())
    c3.metric("Merchants", df["merchant"].nunique())
    c4.metric("Fraud Transactions", int(df["fraud"].sum()))

    st.subheader("Transaction Graph")
    g1, g2 = st.columns(2)
    g1.metric("Graph Nodes", G.number_of_nodes())
    g2.metric("Graph Edges", G.number_of_edges())

    st.subheader("Model / Risk Summary")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Top scaled risk scores")
        top_risk_df = (
            df[["customer", "merchant", "amount", "category", "fraud", "risk_score_scaled"]]
            .sort_values("risk_score_scaled", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        st.dataframe(top_risk_df, use_container_width=True)

    with col2:
        st.write("Risk score statistics")
        risk_stats = df["risk_score_scaled"].describe().to_frame("value")
        st.dataframe(risk_stats, use_container_width=True)

    if run_search:
        st.subheader("Search Scenario")
        st.write(f"**Start:** {node_label(start)}")
        st.write(f"**Goal:** {node_label(goal)}")

        bfs_result = bfs_path(G, start, goal)
        astar_result = astar_path(G, start, goal)

        bfs_risk = path_total_risk(G, bfs_result["path"])
        astar_risk = path_total_risk(G, astar_result["path"])

        st.subheader("Search Comparison")
        a, b, c, d = st.columns(4)
        a.metric("BFS Nodes Expanded", bfs_result["nodes_expanded"])
        b.metric("A* Nodes Expanded", astar_result["nodes_expanded"])
        c.metric("BFS Path Risk", f"{bfs_risk:.6f}")
        d.metric("A* Path Risk", f"{astar_risk:.6f}")

        result_table = pd.DataFrame([
            {
                "Algorithm": "BFS",
                "Path Length": len(bfs_result["path"]) if bfs_result["path"] else None,
                "Nodes Expanded": bfs_result["nodes_expanded"],
                "Runtime (s)": round(bfs_result["runtime"], 6),
                "Total Risk": round(bfs_risk, 6),
                "Total Cost": None
            },
            {
                "Algorithm": "A*",
                "Path Length": len(astar_result["path"]) if astar_result["path"] else None,
                "Nodes Expanded": astar_result["nodes_expanded"],
                "Runtime (s)": round(astar_result["runtime"], 6),
                "Total Risk": round(astar_risk, 6),
                "Total Cost": round(astar_result["total_cost"], 6) if astar_result["total_cost"] is not None else None
            }
        ])

        st.dataframe(result_table, use_container_width=True)

        st.subheader("Visual Comparison")
        chart_col1, chart_col2 = st.columns(2)
        fig_nodes, fig_runtime = make_comparison_chart(
            bfs_result["nodes_expanded"],
            astar_result["nodes_expanded"],
            bfs_result["runtime"],
            astar_result["runtime"]
        )

        with chart_col1:
            st.pyplot(fig_nodes)

        with chart_col2:
            st.pyplot(fig_runtime)

        st.subheader("BFS Path")
        st.code(format_path(bfs_result["path"]), language=None)
        bfs_table = path_to_table(G, bfs_result["path"])
        if not bfs_table.empty:
            st.dataframe(style_fraud_column(bfs_table), use_container_width=True)
        else:
            st.warning("No BFS path found.")

        st.subheader("A* Path")
        st.code(format_path(astar_result["path"]), language=None)
        astar_table = path_to_table(G, astar_result["path"])
        if not astar_table.empty:
            st.dataframe(style_fraud_column(astar_table), use_container_width=True)
        else:
            st.warning("No A* path found.")

    else:
        st.info("Choose a scenario in the sidebar and click 'Run Analysis'.")



with tab_sensitivity:
    st.header("Sensitivity Analysis & Perturbation Tests")
    st.write(
        "Evaluate how robust BFS and A\\* are to changes in the transaction "
        "graph — through edge noise, node removal, risk-threshold shifts, "
        "and full Monte Carlo simulation."
    )

    st.subheader("Active Nodes")
    st.write(f"Using sidebar selection — **Start:** `{start}` | **Goal:** `{goal}`")
    st.caption("Change the node selection in the sidebar to update all experiments.")

    st.divider()


    st.subheader("1 · Edge-Weight Perturbation")
    st.write(
        "Gaussian noise (σ) is added to every edge cost. "
        "Higher σ simulates greater uncertainty in the risk model. "
        "Each noise level is repeated across multiple trials."
    )

    ep_col1, ep_col2 = st.columns(2)
    with ep_col1:
        ep_trials = st.slider("Trials per noise level", 5, 30, 10, key="ep_trials")
    with ep_col2:
        ep_noise_max = st.slider("Max noise σ", 0.10, 0.60, 0.40, step=0.05, key="ep_noise_max")

    if st.button("Run Edge Perturbation", key="btn_ep"):
        noise_levels = [round(v, 3) for v in np.linspace(0.0, ep_noise_max, 7).tolist()]
        with st.spinner("Running edge perturbation trials…"):
            ep_df = edge_perturbation_experiment(G, start, goal, noise_levels, ep_trials)

        st.success(f"Completed {len(ep_df)} trial results.")

        ep_summary = (
            ep_df.groupby(["algorithm", "noise_std"])[["nodes_expanded", "total_risk", "path_length"]]
            .agg(["mean", "std"])
            .round(4)
            .reset_index()
        )
        ep_summary.columns = [" ".join(c).strip() for c in ep_summary.columns]
        st.dataframe(ep_summary, use_container_width=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for algo, grp in ep_df.groupby("algorithm"):
            agg = grp.groupby("noise_std")["nodes_expanded"].agg(["mean", "std"])
            axes[0].errorbar(agg.index, agg["mean"], yerr=agg["std"],
                             label=algo, marker="o", capsize=4)
        axes[0].set_title("Nodes Expanded vs Noise σ")
        axes[0].set_xlabel("Noise σ")
        axes[0].set_ylabel("Nodes Expanded (mean ± std)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        for algo, grp in ep_df.groupby("algorithm"):
            agg = grp.groupby("noise_std")["total_risk"].agg(["mean", "std"])
            axes[1].errorbar(agg.index, agg["mean"], yerr=agg["std"],
                             label=algo, marker="o", capsize=4)
        axes[1].set_title("Total Path Risk vs Noise σ")
        axes[1].set_xlabel("Noise σ")
        axes[1].set_ylabel("Total Risk (mean ± std)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

    st.divider()

    # ----------------------------------------------------------------
    # 2. Node Removal
    # ----------------------------------------------------------------
    st.subheader("2 · Node Removal Perturbation")
    st.write(
        "Nodes whose maximum outgoing risk exceeds a threshold are removed "
        "before search. Simulates blocking high-risk participants from the network."
    )

    nr_thresholds = st.multiselect(
        "Risk thresholds to test",
        options=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        default=[1.0, 0.8, 0.6, 0.4, 0.2],
        key="nr_thresholds"
    )

    if st.button("Run Node Removal", key="btn_nr") and nr_thresholds:
        with st.spinner("Running node removal sweep…"):
            nr_df = node_removal_experiment(G, start, goal,
                                            sorted(nr_thresholds, reverse=True))
        st.success("Done.")
        st.dataframe(nr_df, use_container_width=True)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for algo, grp in nr_df.groupby("algorithm"):
            grp_s = grp.sort_values("risk_threshold")
            axes[0].plot(grp_s["risk_threshold"], grp_s["nodes_removed"], label=algo, marker="o")
        axes[0].set_title("Nodes Removed vs Threshold")
        axes[0].set_xlabel("Risk Threshold")
        axes[0].set_ylabel("Nodes Removed")
        axes[0].invert_xaxis()
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        for algo, grp in nr_df.groupby("algorithm"):
            grp_s = grp.sort_values("risk_threshold")
            axes[1].plot(grp_s["risk_threshold"], grp_s["nodes_expanded"], label=algo, marker="o")
        axes[1].set_title("Nodes Expanded vs Threshold")
        axes[1].set_xlabel("Risk Threshold")
        axes[1].invert_xaxis()
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        for algo, grp in nr_df.groupby("algorithm"):
            grp_s = grp.sort_values("risk_threshold")
            axes[2].plot(grp_s["risk_threshold"], grp_s["path_found"].astype(int),
                         label=algo, marker="o")
        axes[2].set_title("Path Found vs Threshold")
        axes[2].set_xlabel("Risk Threshold")
        axes[2].set_ylabel("Path Found (1 = yes)")
        axes[2].set_ylim(-0.1, 1.1)
        axes[2].invert_xaxis()
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

    st.divider()

    st.subheader("3 · Risk Classification Threshold Sweep")
    st.write(
        "Sweeps the fraud-risk threshold that determines which transactions "
        "get flagged. Shows how the decision boundary affects the paths "
        "each algorithm finds."
    )

    sweep_col1, sweep_col2 = st.columns(2)
    with sweep_col1:
        sweep_min = st.slider("Min threshold", 0.05, 0.50, 0.10, step=0.05, key="sweep_min")
    with sweep_col2:
        sweep_max = st.slider("Max threshold", 0.50, 0.99, 0.95, step=0.05, key="sweep_max")

    if st.button("Run Threshold Sweep", key="btn_sweep"):
        sweep_thresholds = np.linspace(sweep_min, sweep_max, 18).tolist()
        with st.spinner("Rebuilding graphs across threshold values…"):
            sw_df = risk_threshold_sweep(df, start, goal, sweep_thresholds)
        st.success("Done.")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for algo, grp in sw_df.groupby("algorithm"):
            grp_s = grp.sort_values("threshold")
            axes[0].plot(grp_s["threshold"], grp_s["flagged_edges"], label=algo, marker="o")
        axes[0].set_title("Flagged Edges vs Threshold")
        axes[0].set_xlabel("Risk Threshold")
        axes[0].set_ylabel("# Flagged Edges")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        for algo, grp in sw_df.groupby("algorithm"):
            grp_s = grp.sort_values("threshold")
            axes[1].plot(grp_s["threshold"], grp_s["nodes_expanded"], label=algo, marker="o")
        axes[1].set_title("Nodes Expanded vs Threshold")
        axes[1].set_xlabel("Risk Threshold")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        for algo, grp in sw_df.groupby("algorithm"):
            grp_s = grp.sort_values("threshold")
            axes[2].plot(grp_s["threshold"], grp_s["total_risk"], label=algo, marker="o")
        axes[2].set_title("Total Path Risk vs Threshold")
        axes[2].set_xlabel("Risk Threshold")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        st.dataframe(sw_df, use_container_width=True)

    st.divider()

 
    st.subheader("4 · Monte Carlo Perturbation Simulation")
    st.write(
        "Runs many independently-perturbed copies of the graph to build "
        "empirical distributions for nodes expanded, path length, and total risk. "
        "The Coefficient of Variation shows which algorithm is more stable."
    )

    mc_col1, mc_col2 = st.columns(2)
    with mc_col1:
        mc_n = st.slider("Number of simulations", 20, 200, 50, step=10, key="mc_n")
    with mc_col2:
        mc_noise = st.slider("Noise σ per simulation", 0.01, 0.30, 0.05, step=0.01, key="mc_noise")

    if st.button("Run Monte Carlo", key="btn_mc"):
        with st.spinner(f"Running {mc_n} Monte Carlo simulations…"):
            mc_df = monte_carlo_perturbation(G, start, goal, mc_n, mc_noise)
        st.success(f"{mc_n} simulations complete.")

        st.dataframe(monte_carlo_summary(mc_df).round(6), use_container_width=True)

        metrics = [
            ("nodes_expanded", "Nodes Expanded"),
            ("path_length", "Path Length"),
            ("total_risk", "Total Risk"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (col, label) in zip(axes, metrics):
            for algo, grp in mc_df.groupby("algorithm"):
                ax.hist(grp[col].dropna(), bins=20, alpha=0.6, label=algo)
            ax.set_title(f"Distribution: {label}")
            ax.set_xlabel(label)
            ax.set_ylabel("Frequency")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        st.write("**Coefficient of Variation** (std / mean — lower = more stable)")
        cv_rows = []
        for algo, grp in mc_df.groupby("algorithm"):
            cv_rows.append({
                "Algorithm": algo,
                "CV nodes_expanded": round(grp["nodes_expanded"].std() / (grp["nodes_expanded"].mean() + 1e-9), 4),
                "CV path_length": round(grp["path_length"].std() / (grp["path_length"].mean() + 1e-9), 4),
                "CV total_risk": round(grp["total_risk"].std() / (grp["total_risk"].mean() + 1e-9), 4),
            })
        st.dataframe(pd.DataFrame(cv_rows), use_container_width=True)
        
    with tab_agents:
        st.header("Multi-Agent Fraud Analysis")
        st.write("Enter a customer and merchant ID to run the multi-agent pipeline.")

        col1, col2 = st.columns(2)
        with col1:
            agent_customer = st.text_input("Customer ID (e.g. C1093826151)")
        with col2:
            agent_merchant = st.text_input("Merchant ID (e.g. M348934600)")

        if st.button("Run Agent Pipeline"):
            if agent_customer and agent_merchant:
                with st.spinner("Running agents..."):
                    result = agent_graph.invoke({
                        "customer": agent_customer,
                        "merchant": agent_merchant,
                        "graph": G
                    })

                if result.get("error"):
                    st.error(result["error"])
                else:
                    st.subheader("Agent Results")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Risk Score", f"{result['risk_score']:.4f}")
                    col2.metric("Amount", f"${result['amount']:.2f}")
                    col3.metric("Decision", result["decision"])

                    st.info(f"**Reason:** {result['decision_reason']}")
                    st.write("**AI Explanation:**")
                    st.write(result["llm_explanation"])
            else:
                st.warning("Please enter both a customer and merchant ID.")