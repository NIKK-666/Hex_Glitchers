# ============================================================
# EXPORT CELL — paste this as a NEW CELL at the very end of your
# notebook (ba.ipynb / Untitled.ipynb), and run it AFTER Cell 7.
#
# It needs these variables already in memory from earlier cells:
#   model, explainer, sage_model, G, node_list, node_idx, x,
#   edge_index, merged_df, in_degrees, out_degrees, pageranks,
#   features, sage_cols, HIDDEN, EMB_DIM, X_test, device
#
# It fixes two gaps in your current Cell 7 export:
#   1. correlated_data.pkl currently saves raw `merged_df`, which is
#      missing the sage_emb_* / in_degree / out_degree / pagerank
#      columns the model actually trained on. The dashboard would
#      have to guess feature names — now it gets the real, final,
#      fully-scored dataframe instead.
#   2. There was no way for the dashboard to run GraphSAGE itself —
#      only the trained embeddings were saved, not the graph. This
#      adds a graph_snapshot.pkl so the dashboard can do a REAL live
#      forward pass through the GraphSAGE encoder (true inductive
#      scoring), not just replay pre-computed numbers.
# ============================================================
import os, pickle
import numpy as np
import pandas as pd
import torch

DATA_DIR = locals().get("DATA_DIR", "./model_artifacts")
os.makedirs(DATA_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1. Final, frozen forward pass through the trained GraphSAGE model
#    over the FULL graph, so every node gets a clean embedding.
# ------------------------------------------------------------
sage_model.eval()
with torch.no_grad():
    final_sage_emb, _ = sage_model(x, edge_index)
final_sage_emb_np = final_sage_emb.cpu().numpy()

sage_cols_final = [f"sage_emb_{i}" for i in range(final_sage_emb_np.shape[1])]
emb_df = pd.DataFrame(final_sage_emb_np, columns=sage_cols_final, index=node_list)
emb_df.index.name = "txid"

# ------------------------------------------------------------
# 2. Rebuild the definitive scored dataset: one row per transaction,
#    with tabular + graph-topology + GraphSAGE features, and the
#    final hybrid XGBoost risk score attached.
# ------------------------------------------------------------
scored_df = merged_df.drop(columns=sage_cols_final, errors="ignore").merge(
    emb_df, on="txid", how="left"
)
scored_df[sage_cols_final] = scored_df[sage_cols_final].fillna(0.0)
scored_df["in_degree"] = scored_df["txid"].map(in_degrees).fillna(0)
scored_df["out_degree"] = scored_df["txid"].map(out_degrees).fillna(0)
scored_df["pagerank"] = scored_df["txid"].map(pageranks).fillna(0)

# Score with the EXACT feature order the XGBoost model was trained on
scored_df["risk_score"] = model.predict_proba(scored_df[features])[:, 1]

# Mark which transactions were held out of XGBoost training — these are
# the honest stand-in for "a wallet the model has never seen", used by
# the dashboard's live Inductive Triage demo.
held_out_ids = set(X_test.index) if hasattr(X_test, "index") else set()
scored_df["was_held_out"] = scored_df.index.isin(held_out_ids)

scored_df.to_pickle(os.path.join(DATA_DIR, "scored_data.pkl"))

# ------------------------------------------------------------
# 3. Save a lightweight graph snapshot so the dashboard can run a
#    REAL GraphSAGE forward pass at click-time (not a replay).
# ------------------------------------------------------------
graph_snapshot = {
    "node_list": node_list,
    "node_idx": node_idx,
    "x": x.cpu(),
    "edge_index": edge_index.cpu(),
    "in_degrees": in_degrees,
    "out_degrees": out_degrees,
    "pageranks": pageranks,
}
with open(os.path.join(DATA_DIR, "graph_snapshot.pkl"), "wb") as f:
    pickle.dump(graph_snapshot, f)

# ------------------------------------------------------------
# 4. Extend pipeline_metadata with the GraphSAGE constructor shape —
#    a state_dict alone isn't enough to rebuild the model class.
# ------------------------------------------------------------
pipeline_metadata = {
    "features": features,          # full XGBoost feature order (tabular+graph+sage)
    "sage_cols": sage_cols_final,   # which of those are GraphSAGE embedding dims
    "node_list": node_list,
    "in_dim": x.shape[1],
    "hidden": HIDDEN,
    "emb_dim": EMB_DIM,
}
with open(os.path.join(DATA_DIR, "pipeline_metadata.pkl"), "wb") as f:
    pickle.dump(pipeline_metadata, f)

print(f"Exported dashboard-ready bundle to '{DATA_DIR}':")
for fn in ["xgboost_forensics_model.pkl", "shap_explainer.pkl", "graphsage_encoder.pt",
           "pipeline_metadata.pkl", "scored_data.pkl", "graph_snapshot.pkl"]:
    p = os.path.join(DATA_DIR, fn)
    status = "OK" if os.path.exists(p) else "MISSING — rerun Cell 7 first"
    print(f"  [{status}]  {fn}")
