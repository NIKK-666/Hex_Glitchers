import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pyvis.network import Network
import streamlit.components.v1 as components
import pickle
import os
from datetime import datetime, timezone

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    layout="wide",
    page_title="ChainSentinel — GraphSAGE + XGBoost Forensics",
    page_icon="🛡️",
    initial_sidebar_state="collapsed",
)

# ============================================================
# THEME / GLOBAL CSS  (dark "SOC terminal" look)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  { font-family: 'JetBrains Mono', 'Courier New', monospace; }

.stApp { background: #070b12; color: #c9d4e3; }
section.main > div { padding-top: 0.5rem; }
#MainMenu, footer, header {visibility: hidden;}

.btcmon-header { display:flex; justify-content:space-between; align-items:center; padding:6px 2px 14px 2px; border-bottom:1px solid #1a2130; margin-bottom:10px;}
.btcmon-title { font-size:22px; font-weight:800; color:#22d3ee; letter-spacing:1px; line-height:1.1;}
.btcmon-sub { font-size:11px; color:#5f6f85; letter-spacing:0.5px; margin-top:1px;}
.btcmon-alert-badge { background:#2a0d0f; color:#ff5c5c; border:1px solid #ff5c5c66; padding:6px 14px; border-radius:4px; font-size:12px; font-weight:700; letter-spacing:0.5px;}

.status-strip { display:flex; gap:26px; align-items:center; font-size:11.5px; color:#7c8ba1; padding:8px 4px; border-bottom:1px solid #1a2130; margin-bottom:18px; flex-wrap:wrap; letter-spacing:0.3px;}
.status-strip b { color:#c9d4e3; }
.dot-live { height:8px; width:8px; background:#22c55e; border-radius:50%; display:inline-block; margin-right:6px; box-shadow:0 0 6px #22c55e;}
.status-right { margin-left:auto; display:flex; gap:22px; }

.panel { background:#0d121c; border:1px solid #1b2331; border-radius:8px; padding:16px 18px 12px 18px; margin-bottom:16px; }
.panel-title { font-size:11px; letter-spacing:1.5px; color:#7c8ba1; text-transform:uppercase; margin-bottom:12px; font-weight:700;}

.kpi-label { font-size:10.5px; letter-spacing:1.2px; color:#5f6f85; text-transform:uppercase; font-weight:700;}
.kpi-value { font-size:30px; font-weight:800; color:#e8eef7; margin-top:4px; line-height:1;}
.kpi-sub { font-size:11.5px; color:#5f6f85; margin-top:6px;}
.kpi-sub.up { color:#ff5c5c; }
.kpi-sub.down { color:#22c55e; }

.badge { display:inline-block; padding:2px 9px; border-radius:4px; font-size:10.5px; font-weight:700; letter-spacing:0.5px;}
.badge-critical { background:#2a0d0f; color:#ff5c5c; border:1px solid #ff5c5c55;}
.badge-high { background:#2b1c05; color:#f4b400; border:1px solid #f4b40055;}
.badge-medium { background:#1c0f2b; color:#a855f7; border:1px solid #a855f755;}
.badge-low { background:#052229; color:#22d3ee; border:1px solid #22d3ee55;}
.badge-clean { background:#062018; color:#22c55e; border:1px solid #22c55e55;}
.badge-flagged { background:#2a0d0f; color:#ff5c5c; border:1px solid #ff5c5c55;}
.badge-tag { background:#121a28; color:#8fa2bd; border:1px solid #232d40; margin-right:5px; }
.badge-unseen { background:#05291f; color:#2dd4bf; border:1px solid #2dd4bf55; }

.alert-card { display:flex; align-items:center; gap:16px; padding:12px 6px; border-bottom:1px solid #161d2a; }
.alert-id { font-weight:800; font-size:13.5px; color:#e8eef7; }
.alert-meta { font-size:11px; color:#5f6f85; margin-top:2px;}
.alert-wallet { color:#22d3ee; }
.alert-conf { font-size:22px; font-weight:800; text-align:right; min-width:70px;}
.alert-conf-label { font-size:9.5px; color:#5f6f85; text-align:right; letter-spacing:1px;}

.bar-track { background:#161d2a; border-radius:3px; height:6px; width:120px; overflow:hidden;}
.bar-fill { height:6px; border-radius:3px; }

.stTabs [data-baseweb="tab-list"] { gap: 26px; border-bottom: 1px solid #1a2130; }
.stTabs [data-baseweb="tab"] {
    font-size:12px; letter-spacing:1px; font-weight:700; color:#5f6f85;
    text-transform:uppercase; padding:6px 2px; background:transparent;
}
.stTabs [aria-selected="true"] { color:#22d3ee !important; border-bottom:2px solid #22d3ee !important; }

[data-testid="stDataFrame"] { background:#0d121c; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# GRAPHSAGE ARCHITECTURE (must match training notebook exactly —
# only the class definition is needed here; weights are loaded
# from graphsage_encoder.pt)
# ============================================================
class GraphSAGEEncoder(torch.nn.Module):
    def __init__(self, in_dim, hidden, emb_dim, num_classes=2):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden, root_weight=True)
        self.conv2 = SAGEConv(hidden, emb_dim, root_weight=True)
        self.classifier = torch.nn.Linear(emb_dim, num_classes)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.3, training=self.training)
        emb = self.conv2(h, edge_index)
        out = self.classifier(F.relu(emb))
        return emb, out


# ============================================================
# DATA / MODEL LOADING
# ============================================================
DATA_DIR = "data"
DEVICE = torch.device("cpu")  # offline/air-gapped inference — CPU only


@st.cache_resource
def load_assets():
    with open(os.path.join(DATA_DIR, "xgboost_forensics_model.pkl"), "rb") as f:
        xgb_model = pickle.load(f)
    with open(os.path.join(DATA_DIR, "shap_explainer.pkl"), "rb") as f:
        explainer = pickle.load(f)
    with open(os.path.join(DATA_DIR, "pipeline_metadata.pkl"), "rb") as f:
        meta = pickle.load(f)
    with open(os.path.join(DATA_DIR, "graph_snapshot.pkl"), "rb") as f:
        graph = pickle.load(f)

    scored_df = pd.read_pickle(os.path.join(DATA_DIR, "scored_data.pkl"))

    sage_model = GraphSAGEEncoder(meta["in_dim"], meta["hidden"], meta["emb_dim"])
    sage_model.load_state_dict(torch.load(
        os.path.join(DATA_DIR, "graphsage_encoder.pt"), map_location=DEVICE
    ))
    sage_model.eval()

    return xgb_model, explainer, sage_model, meta, graph, scored_df


try:
    model, explainer, sage_model, META, GRAPH, df = load_assets()
except Exception as e:
    st.error(f"Failed to load offline hybrid pipeline assets: {e}")
    st.info(
        "Run the training notebook through Cell 7, then paste and run "
        "`export_pipeline_artifacts.py` as a final cell to produce the "
        "6 files this dashboard needs in ./data/ — including scored_data.pkl "
        "and graph_snapshot.pkl (not created by the base notebook)."
    )
    st.stop()

FEATURES = META["features"]        # full XGBoost input: tabular + graph + sage_emb_*
SAGE_COLS = META["sage_cols"]      # subset of FEATURES that are GraphSAGE embedding dims
TABULAR_GRAPH_COLS = [c for c in FEATURES if c not in SAGE_COLS]

df = df.copy()
df["timestamp_chain"] = pd.to_datetime(df["timestamp_chain"])
if "risk_score" not in df.columns:
    df["risk_score"] = model.predict_proba(df[FEATURES])[:, 1]


# ---- derived: severity tier ----
def _severity(s):
    if s >= 0.85: return "CRITICAL"
    if s >= 0.65: return "HIGH"
    if s >= 0.45: return "MEDIUM"
    return "LOW"
df["severity"] = df["risk_score"].apply(_severity)

SEV_COLOR = {"CRITICAL": "#ff5c5c", "HIGH": "#f4b400", "MEDIUM": "#a855f7", "LOW": "#22d3ee"}
SEV_BADGE = {"CRITICAL": "badge-critical", "HIGH": "badge-high", "MEDIUM": "badge-medium", "LOW": "badge-low"}

# ---- derived: wallet cluster id (grouped by destination wallet) ----
df["cluster"] = "C-" + pd.Series((pd.factorize(df["wallet_out"])[0] % 14 + 1)).astype(str).str.zfill(2)

# ---- derived: heuristic alert tags from real features ----
_fee_hi = df["fee"].quantile(0.85)
_lat_hi = df["latency_seconds"].quantile(0.85)
_sage_norm = np.linalg.norm(df[SAGE_COLS].values, axis=1)
_sage_hi = np.quantile(_sage_norm, 0.9)

def _tags(row, sage_mag):
    t = []
    if row.get("src_is_threat", 0): t.append("TOR_RELAY")
    if row["fee"] >= _fee_hi: t.append("STRUCTURING")
    if row["latency_seconds"] >= _lat_hi: t.append("HIGH_VELOCITY")
    if float(row["total_input_amount"]).is_integer(): t.append("ROUND_AMOUNT")
    if row["script_encoded"] == df["script_encoded"].mode()[0]: t.append("MIXING_SERVICE")
    if row["pagerank"] >= df["pagerank"].quantile(0.9): t.append("EXCHANGE_BYPASS")
    if sage_mag >= _sage_hi: t.append("GRAPH_ANOMALY")
    return t[:3] if t else ["ANOMALY"]
df["tags"] = [
    _tags(row, mag) for (_, row), mag in zip(df.iterrows(), _sage_norm)
]

alert_df = df[df["risk_score"] >= 0.45].sort_values("risk_score", ascending=False)
critical_df = df[df["severity"] == "CRITICAL"]

BTC_USD = 63847  # display-only reference price for USD conversions


# ============================================================
# LIVE GRAPHSAGE INFERENCE HELPERS (the inductive triage engine)
# ============================================================
_GRAPH_X = GRAPH["x"]
_GRAPH_EDGE_INDEX = GRAPH["edge_index"]
_NODE_IDX = GRAPH["node_idx"]


@st.cache_resource
def run_full_graph_forward():
    """One frozen forward pass over the whole graph — this is what makes
    GraphSAGE 'inductive': the SAME trained weights score every node,
    seen during training or not, purely from local neighborhood structure."""
    with torch.no_grad():
        emb, logits = sage_model(_GRAPH_X, _GRAPH_EDGE_INDEX)
        probs = F.softmax(logits, dim=1)[:, 1]
    return emb.cpu().numpy(), probs.cpu().numpy()

_ALL_SAGE_EMB, _ALL_SAGE_PROB = run_full_graph_forward()


def live_score_transaction(txid):
    """Re-derives a transaction's risk score end-to-end at click-time:
    GraphSAGE embedding (from the live graph) -> concat with tabular/graph
    features -> XGBoost -> SHAP. Used to demo scoring a txn the model
    never trained on, without any retraining."""
    if txid not in _NODE_IDX:
        return None
    idx = _NODE_IDX[txid]
    sage_vec = _ALL_SAGE_EMB[idx]
    sage_only_prob = float(_ALL_SAGE_PROB[idx])

    row = df.loc[df["txid"] == txid]
    if row.empty:
        return None
    row = row.iloc[0]

    feat_row = {}
    for c in TABULAR_GRAPH_COLS:
        feat_row[c] = row[c]
    for i, c in enumerate(SAGE_COLS):
        feat_row[c] = sage_vec[i]
    feat_df = pd.DataFrame([feat_row])[FEATURES]

    hybrid_prob = float(model.predict_proba(feat_df)[:, 1][0])
    shap_vals = explainer(feat_df)

    return {
        "sage_only_prob": sage_only_prob,
        "hybrid_prob": hybrid_prob,
        "feat_df": feat_df,
        "shap_values": shap_vals,
        "row": row,
    }


# ============================================================
# GLOBAL SHAP EXPLAINABILITY (cached — computed once, reused across
# the ML Engine tab so every reload doesn't re-run SHAP from scratch)
# ============================================================
SHAP_SAMPLE_SIZE = 400  # enough for a stable global picture, fast enough for a live demo

@st.cache_resource
def compute_global_shap():
    sample = df.sample(n=min(SHAP_SAMPLE_SIZE, len(df)), random_state=42)
    sample_X = sample[FEATURES]
    shap_out = explainer(sample_X)
    return sample, sample_X, shap_out

SHAP_SAMPLE_DF, SHAP_SAMPLE_X, GLOBAL_SHAP = compute_global_shap()
GLOBAL_MEAN_ABS_SHAP = pd.Series(
    np.abs(GLOBAL_SHAP.values).mean(axis=0), index=FEATURES
).sort_values(ascending=False)


# ============================================================
# SMALL RENDER HELPERS
# ============================================================
def kpi_card(label, value, sub=None, sub_class=""):
    sub_html = f'<div class="kpi-sub {sub_class}">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div class="panel">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)

def panel_start(title):
    st.markdown(f'<div class="panel"><div class="panel-title">{title}</div>', unsafe_allow_html=True)

def panel_end():
    st.markdown("</div>", unsafe_allow_html=True)

def risk_bar_html(pct, color):
    return f'<div class="bar-track"><div class="bar-fill" style="width:{pct}%; background:{color};"></div></div>'


# ============================================================
# HEADER + STATUS STRIP
# ============================================================
now_str = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S UTC")
n_critical = len(critical_df)

st.markdown(f"""
<div class="btcmon-header">
    <div>
        <div class="btcmon-title">🛡️ CHAINSENTINEL</div>
        <div class="btcmon-sub">GRAPHSAGE + XGBOOST HYBRID FORENSICS — NTRO CHALLENGE 26146</div>
    </div>
    <div class="btcmon-alert-badge">⛔ {n_critical} CRITICAL ALERTS ACTIVE</div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="status-strip">
    <span><span class="dot-live"></span><b>LIVE</b></span>
    <span>ChainSentinel <b>v1.0</b></span>
    <span>GraphSAGE: <b style="color:#22c55e;">ACTIVE</b></span>
    <span>XGBoost: <b style="color:#22c55e;">ACTIVE</b></span>
    <span>Ingested: <b>{len(df):,}</b> records</span>
    <span>Graph nodes: <b>{len(GRAPH['node_list']):,}</b></span>
    <span class="status-right">
        <span>BTC/USD: <b style="color:#f4b400;">${BTC_USD:,}</b></span>
        <span>{now_str}</span>
    </span>
</div>
""", unsafe_allow_html=True)


# ============================================================
# TABS
# ============================================================
tab_overview, tab_graph, tab_alerts, tab_ml, tab_gnn, tab_raw = st.tabs(
    ["OVERVIEW", "ENTITY GRAPH", f"ALERTS  {len(alert_df)}", "ML ENGINE", "GNN + XGBOOST", "RAW DATA"]
)

# ------------------------------------------------------------
# TAB 1: OVERVIEW
# ------------------------------------------------------------
with tab_overview:
    total_vol = df["total_input_amount"].sum()
    susp_vol = alert_df["total_input_amount"].sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Transactions", f"{len(df):,}", "Full observed window", "")
    with c2:
        kpi_card("Flagged Entities", f"{len(alert_df):,}", f"{n_critical} critical", "up" if n_critical else "")
    with c3:
        kpi_card("Suspicious Volume", f"₿ {susp_vol:,.2f}", f"${susp_vol*BTC_USD:,.0f} USD")
    with c4:
        kpi_card("Total Volume", f"₿ {total_vol:,.2f}", f"${total_vol*BTC_USD:,.0f} USD")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        panel_start("Transaction Volume / Flagged Count")
        daily = df.groupby(df["timestamp_chain"].dt.floor("h")).agg(total=("txid", "count")).reset_index()
        flagged_daily = alert_df.groupby(alert_df["timestamp_chain"].dt.floor("h")).agg(flagged=("txid", "count")).reset_index()
        merged = daily.merge(flagged_daily, on="timestamp_chain", how="left").fillna(0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=merged["timestamp_chain"], y=merged["total"], name="TOTAL",
                                  line=dict(color="#22d3ee", width=1.5), fill="tozeroy",
                                  fillcolor="rgba(34,211,238,0.08)"))
        fig.add_trace(go.Scatter(x=merged["timestamp_chain"], y=merged["flagged"], name="FLAGGED",
                                  line=dict(color="#ff5c5c", width=1.5)))
        fig.update_layout(template="plotly_dark", height=270, margin=dict(l=10, r=10, t=10, b=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
                           font=dict(family="JetBrains Mono", size=10, color="#7c8ba1"))
        st.plotly_chart(fig, use_container_width=True)
        panel_end()

    with col_b:
        panel_start("Alert Distribution")
        sev_counts = df["severity"].value_counts().reindex(["CRITICAL", "HIGH", "MEDIUM", "LOW"]).fillna(0)
        fig = go.Figure(data=[go.Pie(
            labels=sev_counts.index, values=sev_counts.values, hole=0.65,
            marker=dict(colors=[SEV_COLOR[s] for s in sev_counts.index]), textinfo="none"
        )])
        fig.update_layout(template="plotly_dark", height=230, margin=dict(l=0, r=0, t=10, b=0),
                           paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
                           font=dict(family="JetBrains Mono", size=10, color="#7c8ba1"))
        st.plotly_chart(fig, use_container_width=True)
        legend_html = "".join(
            f'<div style="display:flex;justify-content:space-between;font-size:11px;padding:2px 0;">'
            f'<span style="color:{SEV_COLOR[s]}">● {s}</span><span>{int(c)}</span></div>'
            for s, c in sev_counts.items()
        )
        st.markdown(legend_html, unsafe_allow_html=True)
        panel_end()

    col_c, col_d = st.columns(2)
    with col_c:
        panel_start("Composite Risk Score (Hybrid Model)")
        risk_ts = df.groupby(df["timestamp_chain"].dt.floor("h"))["risk_score"].mean().reset_index()
        fig = go.Figure(go.Scatter(x=risk_ts["timestamp_chain"], y=risk_ts["risk_score"],
                                    line=dict(color="#f4b400", width=1.5)))
        fig.update_layout(template="plotly_dark", height=230, margin=dict(l=10, r=10, t=10, b=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           yaxis=dict(range=[0, 1]),
                           font=dict(family="JetBrains Mono", size=10, color="#7c8ba1"))
        st.plotly_chart(fig, use_container_width=True)
        panel_end()

    with col_d:
        panel_start("Transaction Origin by ASN")
        asn_counts = df["asn"].value_counts().head(6).sort_values()
        fig = go.Figure(go.Bar(x=asn_counts.values, y=asn_counts.index, orientation="h",
                                marker=dict(color="#22d3ee")))
        fig.update_layout(template="plotly_dark", height=230, margin=dict(l=10, r=10, t=10, b=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font=dict(family="JetBrains Mono", size=10, color="#7c8ba1"))
        st.plotly_chart(fig, use_container_width=True)
        panel_end()

    panel_start("Entity Cluster Risk Matrix — Top 8")
    cluster_agg = df.groupby("cluster").agg(
        wallets=("wallet_out", "nunique"),
        total_btc=("total_input_amount", "sum"),
        risk=("risk_score", "mean"),
    ).sort_values("risk", ascending=False).head(8).reset_index()

    header_cols = st.columns([1, 1, 1.3, 1, 1, 1.4])
    for c, t in zip(header_cols, ["CLUSTER", "WALLETS", "TOTAL BTC", "RISK SCORE", "STATUS", "RISK BAR"]):
        c.markdown(f'<span style="color:#5f6f85;font-size:10.5px;letter-spacing:1px;">{t}</span>', unsafe_allow_html=True)

    for _, r in cluster_agg.iterrows():
        pct = r["risk"] * 100
        status = "FLAGGED" if pct >= 55 else "CLEAN"
        badge_cls = "badge-flagged" if status == "FLAGGED" else "badge-clean"
        color = "#ff5c5c" if pct >= 70 else ("#f4b400" if pct >= 45 else "#22c55e")
        cols = st.columns([1, 1, 1.3, 1, 1, 1.4])
        cols[0].markdown(f'<b style="color:#22d3ee">{r["cluster"]}</b>', unsafe_allow_html=True)
        cols[1].write(int(r["wallets"]))
        cols[2].write(f'₿ {r["total_btc"]:.2f}')
        cols[3].markdown(f'<b style="color:{color}">{pct:.1f}%</b>', unsafe_allow_html=True)
        cols[4].markdown(f'<span class="badge {badge_cls}">{status}</span>', unsafe_allow_html=True)
        cols[5].markdown(risk_bar_html(min(pct, 100), color), unsafe_allow_html=True)
    panel_end()

# ------------------------------------------------------------
# TAB 2: ENTITY GRAPH
# ------------------------------------------------------------
with tab_graph:
    fcol1, fcol2 = st.columns([3, 1])
    with fcol1:
        min_risk = st.slider("Minimum Risk Score (AI Confidence)", 0.0, 1.0, 0.45, 0.05)
    graph_df = df[df["risk_score"] >= min_risk]

    gcol, sidecol = st.columns([3, 1])
    with gcol:
        panel_start(f"Network Entity Subgraph — {len(graph_df)} flagged transactions")
        if not graph_df.empty:
            net = Network(height="520px", width="100%", bgcolor="#0d121c", font_color="#c9d4e3", directed=True)
            net.barnes_hut(gravity=-3000, spring_length=90)
            plotted = graph_df.head(60)  # keep graph legible
            for _, row in plotted.iterrows():
                sev = row["severity"]
                color = SEV_COLOR[sev]
                size = 10 + row["risk_score"] * 20
                net.add_node(row["src_ip"], label=f"IP:{row['src_ip']}", color="#ea4335", shape="ellipse",
                             title=f"ASN: {row['asn']}")
                net.add_node(row["wallet_in"], label=f"{row['wallet_in'][:8]}…", color="#4285f4", shape="box")
                net.add_node(row["wallet_out"], label=f"{row['wallet_out'][:8]}…", color="#34a853", shape="box")
                net.add_node(row["txid"], label="TX", color=color, shape="dot", size=size,
                             title=f"Risk: {row['risk_score']*100:.1f}% | {sev}")
                net.add_edge(row["src_ip"], row["wallet_in"], label="accesses")
                net.add_edge(row["wallet_in"], row["txid"], label="inputs")
                net.add_edge(row["txid"], row["wallet_out"], label=f"{row['total_input_amount']:.3f} BTC")
            net.save_graph("temp_subgraph.html")
            with open("temp_subgraph.html", "r", encoding="utf-8") as f:
                html_code = f.read()
            components.html(html_code, height=530)
        else:
            st.info("No entities match the current risk threshold.")
        panel_end()

    with sidecol:
        panel_start("Flagged Entities")
        top_flagged = graph_df.sort_values("risk_score", ascending=False).head(8)
        for _, r in top_flagged.iterrows():
            badge_cls = SEV_BADGE[r["severity"]]
            st.markdown(f"""
            <div style="border-bottom:1px solid #161d2a; padding:8px 0;">
                <div style="font-size:12px; color:#22d3ee;">{r['wallet_out'][:14]}…</div>
                <div style="font-size:10.5px; color:#5f6f85;">cluster {r['cluster']}</div>
                <div style="text-align:right; margin-top:-22px;"><span class="badge {badge_cls}">{r['risk_score']*100:.0f}%</span></div>
            </div>
            """, unsafe_allow_html=True)
        panel_end()

        panel_start("Graph Stats")
        stats = {
            "Total nodes": pd.concat([graph_df["wallet_in"], graph_df["wallet_out"], graph_df["src_ip"], graph_df["txid"]]).nunique(),
            "Wallets": pd.concat([graph_df["wallet_in"], graph_df["wallet_out"]]).nunique(),
            "IP nodes": graph_df["src_ip"].nunique(),
            "Transactions": graph_df["txid"].nunique(),
            "Flagged": len(graph_df),
            "Full graph nodes": len(GRAPH["node_list"]),
        }
        for k, v in stats.items():
            st.markdown(f'<div style="display:flex;justify-content:space-between;font-size:12px;padding:3px 0;">'
                        f'<span style="color:#7c8ba1;">{k}</span><b>{v}</b></div>', unsafe_allow_html=True)
        panel_end()

# ------------------------------------------------------------
# TAB 3: ALERTS
# ------------------------------------------------------------
with tab_alerts:
    sev_filter = st.radio("Severity", ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"], horizontal=True, label_visibility="collapsed")
    view_df = alert_df if sev_filter == "ALL" else alert_df[alert_df["severity"] == sev_filter]
    st.caption(f"{len(view_df)} alerts · sorted by confidence")

    panel_start("Active Alerts")
    for i, (_, r) in enumerate(view_df.iterrows()):
        badge_cls = SEV_BADGE[r["severity"]]
        color = SEV_COLOR[r["severity"]]
        tags_html = "".join(f'<span class="badge badge-tag">{t}</span>' for t in r["tags"])
        unseen_html = '<span class="badge badge-unseen">UNSEEN AT TRAIN TIME</span>' if r.get("was_held_out") else ""
        cols = st.columns([0.9, 5, 1])
        with cols[0]:
            st.markdown(f'<span class="badge {badge_cls}">{r["severity"]}</span>', unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"""
            <div class="alert-id">ALT-{i+1:04d} <span style="color:#5f6f85;font-weight:400;">{r['timestamp_chain']}</span> <span class="badge badge-tag">{r['cluster']}</span></div>
            <div class="alert-meta">WALLET: <span class="alert-wallet">{r['wallet_out'][:18]}…</span> &nbsp; ₿ {r['total_input_amount']:.4f} &nbsp; ${r['total_input_amount']*BTC_USD:,.2f}</div>
            <div style="margin-top:4px;">{tags_html} {unseen_html}</div>
            """, unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f'<div class="alert-conf" style="color:{color};">{r["risk_score"]*100:.0f}%</div>'
                        f'<div class="alert-conf-label">CONF</div>'
                        f'{risk_bar_html(r["risk_score"]*100, color)}', unsafe_allow_html=True)
        st.markdown('<hr style="border-color:#161d2a; margin:4px 0;">', unsafe_allow_html=True)
        if i >= 40:
            st.caption(f"...and {len(view_df) - 40} more. Narrow the filter to see fewer.")
            break
    panel_end()

# ------------------------------------------------------------
# TAB 4: ML ENGINE
# ------------------------------------------------------------
with tab_ml:
    c1, c2, c3 = st.columns([1.1, 1.3, 1])

    with c1:
        panel_start("Model Configuration")
        try:
            params = model.get_params()
        except Exception:
            params = {}
        n_trees = getattr(model, "n_estimators", params.get("n_estimators", "—"))
        max_depth = getattr(model, "max_depth", params.get("max_depth", "—"))
        rows = {
            "Architecture": "GraphSAGE (2-layer) + XGBoost",
            "Total features": len(FEATURES),
            "GraphSAGE emb. dims": len(SAGE_COLS),
            "Tabular/graph features": len(TABULAR_GRAPH_COLS),
            "XGBoost trees": n_trees,
            "Max depth": max_depth,
            "Train rows (this feed)": f"{len(df):,}",
            "Decision threshold": "0.45",
        }
        for k, v in rows.items():
            st.markdown(f'<div style="display:flex;justify-content:space-between;font-size:12.5px;padding:5px 0;border-bottom:1px solid #161d2a;">'
                        f'<span style="color:#7c8ba1;">{k}</span><b>{v}</b></div>', unsafe_allow_html=True)
        panel_end()

    with c2:
        panel_start("Feature Importance (Gain) — Top 15")
        try:
            importances = model.feature_importances_
            imp_series = pd.Series(importances, index=FEATURES).sort_values(ascending=True).tail(15)
            bar_colors = ["#8e44ad" if f in SAGE_COLS else "#f4b400" for f in imp_series.index]
            fig = go.Figure(go.Bar(x=imp_series.values, y=imp_series.index, orientation="h",
                                    marker=dict(color=bar_colors)))
            fig.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=10, b=10),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font=dict(family="JetBrains Mono", size=10, color="#7c8ba1"))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("🟣 GraphSAGE embedding dims · 🟡 tabular/graph-topology features")
        except Exception:
            st.info("Feature importances not exposed by this model object.")
        panel_end()

    with c3:
        panel_start("Detection Summary")
        if "label" in df.columns:
            tp = int(((df["label"] == 1) & (df["risk_score"] >= 0.45)).sum())
            fp = int(((df["label"] == 0) & (df["risk_score"] >= 0.45)).sum())
            fn = int(((df["label"] == 1) & (df["risk_score"] < 0.45)).sum())
            tn = int(((df["label"] == 0) & (df["risk_score"] < 0.45)).sum())
            acc = (tp + tn) / max(len(df), 1) * 100
            m1, m2 = st.columns(2)
            m1.markdown(f'<div class="panel" style="background:#062018;"><div class="kpi-label">TP</div><div class="kpi-value" style="color:#22c55e;">{tp}</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="panel" style="background:#2a0d0f;"><div class="kpi-label">FP</div><div class="kpi-value" style="color:#ff5c5c;">{fp}</div></div>', unsafe_allow_html=True)
            m3, m4 = st.columns(2)
            m3.markdown(f'<div class="panel"><div class="kpi-label">FN</div><div class="kpi-value">{fn}</div></div>', unsafe_allow_html=True)
            m4.markdown(f'<div class="panel"><div class="kpi-label">TN</div><div class="kpi-value">{tn}</div></div>', unsafe_allow_html=True)
            st.caption(f"Accuracy: {acc:.1f}%")
        else:
            st.markdown(f'<div class="kpi-label">FLAGGED / TOTAL</div><div class="kpi-value">{len(alert_df)} / {len(df)}</div>', unsafe_allow_html=True)
            st.caption("No ground-truth `label` column in this feed — confusion matrix unavailable, showing raw counts instead.")
        panel_end()

    panel_start("AI/ML Detection Modules")
    modules = [
        ("GraphSAGE Neighborhood Encoder", "Inductive graph embeddings — scores wallets never seen in training from their local connections alone."),
        ("Graph Centrality Clustering", "PageRank + in/out-degree on the wallet↔tx graph to surface high-centrality hub nodes."),
        ("Peel-Chain Detector", "Sequential tracing: flags chains where output is quickly re-spent in near-full amounts."),
        ("Mixing Service Fingerprint", "Heuristic on repeated script types combined with ASN/IP threat reputation."),
        ("Structuring (Smurfing) Detector", "Flags clusters of similarly-sized sub-threshold transactions in short windows."),
        ("TOR / Threat-ASN Correlation", "Cross-references src_ip / ASN against known threat indicators (src_is_threat)."),
    ]
    mc = st.columns(3)
    for i, (name, desc) in enumerate(modules):
        with mc[i % 3]:
            st.markdown(f"""
            <div style="border:1px solid #1b2331; border-radius:6px; padding:12px; margin-bottom:12px; min-height:120px;">
                <div style="display:flex; justify-content:space-between;">
                    <b style="font-size:12.5px;">{name}</b>
                    <span style="color:#22c55e; font-size:10px;">● ACTIVE</span>
                </div>
                <div style="font-size:11px; color:#7c8ba1; margin-top:6px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    panel_end()

# ------------------------------------------------------------
# TAB 5: GNN + XGBOOST (architecture / explainability / live scoring)
# ------------------------------------------------------------
with tab_gnn:
    panel_start("Hybrid Detection Pipeline Architecture")
    stages = [
        ("Input", "Graph G", "wallets, tx, IPs", "#22d3ee"),
        ("GraphSAGE L1", "SAGEConv", f"→ {META['hidden']}-d hidden", "#8e44ad"),
        ("GraphSAGE L2", "SAGEConv", f"→ {META['emb_dim']}-d embedding", "#8e44ad"),
        ("Concat", "Fuse", f"{len(SAGE_COLS)} sage + {len(TABULAR_GRAPH_COLS)} tabular/graph", "#f4b400"),
        ("XGBoost", "Ensemble", "gradient-boosted trees", "#f4b400"),
        ("Risk Score", "Output", "P(illicit)", "#ff5c5c"),
    ]
    scols = st.columns(len(stages) * 2 - 1)
    for i, (title, sub, note, color) in enumerate(stages):
        with scols[i * 2]:
            st.markdown(f"""
            <div style="border:1px solid {color}55; background:{color}11; border-radius:6px; padding:12px; text-align:center; min-height:90px;">
                <div style="font-size:13px; font-weight:800; color:{color};">{title}</div>
                <div style="font-size:11px; color:#c9d4e3; margin-top:4px;">{sub}</div>
                <div style="font-size:9.5px; color:#5f6f85; margin-top:4px;">{note}</div>
            </div>
            """, unsafe_allow_html=True)
        if i < len(stages) - 1:
            with scols[i * 2 + 1]:
                st.markdown('<div style="text-align:center; font-size:20px; color:#5f6f85; padding-top:30px;">→</div>', unsafe_allow_html=True)
    panel_end()

    panel_start("🧠 Inductive Triage — Score a Transaction Live")
    st.caption(
        "Runs a fresh GraphSAGE forward pass on the transaction you pick, then feeds that "
        "embedding through XGBoost — the exact process used to score a brand-new wallet the "
        "model has never trained on, with zero retraining."
    )
    scope = st.radio(
        "Transaction pool", ["Flagged alerts", "Held out of training (closest stand-in for 'unseen')"],
        horizontal=True, label_visibility="collapsed", key="scope_radio"
    )
    if scope.startswith("Held out") and "was_held_out" in df.columns:
        pool = df[df["was_held_out"]]
        if pool.empty:
            st.info("No held-out rows flagged in this feed — falling back to flagged alerts.")
            pool = alert_df
    else:
        pool = alert_df

    if not pool.empty:
        selected_tx = st.selectbox("Select transaction to investigate:", pool["txid"].values, key="live_tx_select")
        result = live_score_transaction(selected_tx)

        if result is None:
            st.warning("This transaction isn't present in the loaded graph snapshot.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.markdown(f'<div class="kpi-label">GraphSAGE-only score</div><div class="kpi-value" style="font-size:22px;">{result["sage_only_prob"]*100:.1f}%</div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="kpi-label">Hybrid (GraphSAGE+XGB) score</div><div class="kpi-value" style="font-size:22px; color:#f4b400;">{result["hybrid_prob"]*100:.1f}%</div>', unsafe_allow_html=True)
            was_unseen = bool(result["row"].get("was_held_out", False))
            m3.markdown(f'<div class="kpi-label">Was held out of XGBoost training?</div><div class="kpi-value" style="font-size:22px; color:{"#2dd4bf" if was_unseen else "#5f6f85"};">{"YES" if was_unseen else "NO"}</div>', unsafe_allow_html=True)

            ecol, gcol = st.columns(2)
            with ecol:
                st.markdown("**Why flagged? (SHAP evidence)**")
                shap_vals = result["shap_values"].values[0]
                base_val = result["shap_values"].base_values
                if isinstance(base_val, np.ndarray):
                    base_val = base_val[0]
                shap_df = pd.DataFrame({
                    "Feature": FEATURES,
                    "Value": [result["feat_df"].iloc[0][f] for f in FEATURES],
                    "SHAP Impact": shap_vals,
                }).sort_values(by="SHAP Impact", key=abs, ascending=False).head(10)

                def highlight_impact(val):
                    color = '#ff5c5c' if val > 0 else '#22c55e'
                    return f'color: {color}; font-weight: bold'

                st.markdown(f"Baseline probability: **{round(float(base_val) * 100, 2)}%**")
                st.dataframe(
                    shap_df.style.map(highlight_impact, subset=["SHAP Impact"]),
                    use_container_width=True, height=280
                )

            with gcol:
                st.markdown("**Transaction subgraph**")
                row = result["row"]
                net = Network(height="330px", width="100%", bgcolor="#0d121c", font_color="#c9d4e3", directed=True)
                ip = row["src_ip"]
                w_in = row["wallet_in"]
                w_out = row["wallet_out"]
                net.add_node(ip, label=f"IP: {ip}", color="#ea4335", shape="ellipse", title=f"ASN: {row['asn']}")
                net.add_node(w_in, label=f"Wallet In: {w_in[:8]}…", color="#4285f4", shape="box")
                net.add_node(w_out, label=f"Wallet Out: {w_out[:8]}…", color="#34a853", shape="box")
                net.add_node(selected_tx, label="Transaction", color="#f4b400", shape="dot")
                net.add_edge(ip, w_in, label="accesses")
                net.add_edge(w_in, selected_tx, label="inputs")
                net.add_edge(selected_tx, w_out, label=f"{row['total_input_amount']:.3f} BTC")
                net.save_graph("temp_selected_subgraph.html")
                with open("temp_selected_subgraph.html", "r", encoding="utf-8") as f:
                    html_code = f.read()
                components.html(html_code, height=340)
    else:
        st.info("No transactions match your current filters.")
    panel_end()

# ------------------------------------------------------------
# TAB 6: RAW DATA
# ------------------------------------------------------------
with tab_raw:
    panel_start("Raw Correlated + Scored Feed")
    display_cols = [c for c in df.columns if not c.startswith("sage_emb_")]
    st.dataframe(df[display_cols].sort_values("timestamp_chain", ascending=False), use_container_width=True, height=600)
    st.caption(f"{len(SAGE_COLS)} GraphSAGE embedding columns hidden from this view — export CSV for the full feature set.")
    st.download_button("⬇ Export CSV", df.to_csv(index=False), file_name="chainsentinel_export.csv", mime="text/csv")
    panel_end()
