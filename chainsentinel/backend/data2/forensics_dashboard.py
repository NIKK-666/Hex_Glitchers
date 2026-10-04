import os
import pickle
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn.functional as F
from pyvis.network import Network
from torch_geometric.nn import SAGEConv

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    layout="wide",
    page_title="ChainSentinel // Intelligent Bitcoin Forensics",
    page_icon="🛡️",
    initial_sidebar_state="expanded",
)

# ============================================================
# PREMIUM STITCHER-INSPIRED DARK THEME (CSS)
# ============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

.stApp {
    background: radial-gradient(circle at 50% 0%, #0f172a 0%, #070b12 100%);
    color: #e2e8f0;
}
section.main > div { padding-top: 1rem; padding-bottom: 2rem; }
#MainMenu, footer, header { visibility: hidden; }

/* ---------- Header ---------- */
.stitcher-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 24px;
    background: rgba(13, 18, 28, 0.45);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
}
.stitcher-title {
    font-size: 24px;
    font-weight: 800;
    background: linear-gradient(90deg, #00F2FE 0%, #4FACFE 50%, #9B51E0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
    line-height: 1.2;
}
.stitcher-sub {
    font-size: 12px;
    color: #64748b;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 4px;
}
.stitcher-status-badge {
    background: rgba(0, 242, 254, 0.1);
    color: #00F2FE;
    border: 1px solid rgba(0, 242, 254, 0.3);
    padding: 6px 16px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}

/* ---------- Status strip ---------- */
.status-strip {
    display: flex;
    gap: 32px;
    align-items: center;
    font-size: 12px;
    color: #94a3b8;
    padding: 12px 20px;
    background: rgba(13, 18, 28, 0.2);
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.02);
    margin-bottom: 20px;
    flex-wrap: wrap;
}
.status-strip b { color: #f8fafc; font-weight: 600; }
.dot-live {
    height: 8px; width: 8px; background: #00F2FE; border-radius: 50%;
    display: inline-block; margin-right: 8px; box-shadow: 0 0 10px #00F2FE;
    animation: pulse 1.6s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(0,242,254,0.5); }
    70% { box-shadow: 0 0 0 8px rgba(0,242,254,0); }
    100% { box-shadow: 0 0 0 0 rgba(0,242,254,0); }
}

/* ---------- Glass panels ---------- */
.panel {
    background: rgba(13, 18, 28, 0.7);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.panel:hover { border-color: rgba(0, 242, 254, 0.15); }
.panel-title {
    font-size: 12px; letter-spacing: 1.5px; color: #94a3b8;
    text-transform: uppercase; margin-bottom: 16px; font-weight: 700;
}

/* ---------- KPI cards ---------- */
.kpi-card {
    background: rgba(13, 18, 28, 0.7);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.kpi-card:hover { transform: translateY(-3px); border-color: rgba(0,242,254,0.25); }
.kpi-label { font-size: 11px; letter-spacing: 1px; color: #64748b; text-transform: uppercase; font-weight: 700; }
.kpi-value { font-size: 32px; font-weight: 800; color: #ffffff; margin-top: 6px; line-height: 1.1; letter-spacing: -1px; }
.kpi-sub { font-size: 12px; color: #64748b; margin-top: 8px; font-family: 'JetBrains Mono', monospace; }
.kpi-sub.up { color: #ef4444; }
.kpi-sub.down { color: #10b981; }

/* ---------- Badges ---------- */
.badge { display: inline-block; padding: 3px 10px; border-radius: 6px; font-size: 11px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.badge-critical { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }
.badge-high { background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
.badge-medium { background: rgba(155, 81, 224, 0.15); color: #c084fc; border: 1px solid rgba(155, 81, 224, 0.3); }
.badge-low { background: rgba(0, 242, 254, 0.15); color: #00F2FE; border: 1px solid rgba(0, 242, 254, 0.3); }
.badge-clean { background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
.badge-flagged { background: rgba(239, 68, 68, 0.12); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.25); }
.badge-tag { background: #1e293b; color: #94a3b8; border: 1px solid #334155; margin-right: 4px; }
.badge-unseen { background: rgba(6, 182, 212, 0.15); color: #22d3ee; border: 1px solid rgba(6, 182, 212, 0.3); }

/* ---------- Risk bars ---------- */
.risk-track { width: 100%; height: 6px; background: rgba(255,255,255,0.06); border-radius: 4px; overflow: hidden; margin-top: 6px; }
.risk-fill { height: 100%; border-radius: 4px; }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 16px; border-bottom: 1px solid rgba(255, 255, 255, 0.05); padding-left: 8px; }
.stTabs [data-baseweb="tab"] {
    font-size: 13px; letter-spacing: 0.5px; font-weight: 600; color: #64748b;
    padding: 10px 16px; background: transparent; border-radius: 6px 6px 0 0; transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover { color: #f8fafc; background: rgba(255, 255, 255, 0.02); }
.stTabs [aria-selected="true"] {
    color: #00F2FE !important; border-bottom: 2px solid #00F2FE !important; background: rgba(0, 242, 254, 0.04) !important;
}

/* ---------- Dataframe ---------- */
[data-testid="stDataFrame"] { background: #0d121c; border-radius: 10px; border: 1px solid rgba(255,255,255,0.05); }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background: #0a0e17; border-right: 1px solid rgba(255,255,255,0.05); }
[data-testid="stSidebar"] .stMarkdown p { color: #94a3b8; }

/* ---------- Expander (alert rows) ---------- */
[data-testid="stExpander"] {
    background: rgba(13, 18, 28, 0.55);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 10px;
    margin-bottom: 8px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# GRAPHSAGE ARCHITECTURE DEFINITION
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
# ASSET LOADING (cached, silent warnings)
# ============================================================
DATA_DIR = "data2"
DEVICE = torch.device("cpu")


@st.cache_resource
def load_assets():
    # ADD THESE THREE LINES HERE TO MUTE THE WARNINGS:
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
    warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

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
    st.stop()

# ============================================================
# DATA PIPELINE
# ============================================================
FEATURES = META["features"]
SAGE_COLS = META["sage_cols"]
TABULAR_GRAPH_COLS = [c for c in FEATURES if c not in SAGE_COLS]

df = df.copy()
df["timestamp_chain"] = pd.to_datetime(df["timestamp_chain"])
if "risk_score" not in df.columns:
    df["risk_score"] = model.predict_proba(df[FEATURES])[:, 1]


def _severity(s):
    if s >= 0.85:
        return "CRITICAL"
    if s >= 0.65:
        return "HIGH"
    if s >= 0.45:
        return "MEDIUM"
    return "LOW"


df["severity"] = df["risk_score"].apply(_severity)

SEV_COLOR = {"CRITICAL": "#ef4444", "HIGH": "#f59e0b", "MEDIUM": "#9B51E0", "LOW": "#00F2FE"}
SEV_BADGE = {"CRITICAL": "badge-critical", "HIGH": "badge-high", "MEDIUM": "badge-medium", "LOW": "badge-low"}

df["cluster"] = "C-" + (pd.factorize(df["wallet_out"])[0] % 14 + 1).astype(str).str.zfill(2)

_fee_hi = df["fee"].quantile(0.85)
_lat_hi = df["latency_seconds"].quantile(0.85)
_sage_norm = np.linalg.norm(df[SAGE_COLS].values, axis=1)
_sage_hi = np.quantile(_sage_norm, 0.9)
_script_mode = df["script_encoded"].mode().iloc[0]


def _tags(row, sage_mag):
    t = []
    if row.get("src_is_threat", 0):
        t.append("TOR_RELAY")
    if row["fee"] >= _fee_hi:
        t.append("STRUCTURING")
    if row["latency_seconds"] >= _lat_hi:
        t.append("HIGH_VELOCITY")
    if float(row["total_input_amount"]).is_integer():
        t.append("ROUND_AMOUNT")
    if row["script_encoded"] == _script_mode:
        t.append("MIXING_SERVICE")
    if row["pagerank"] >= df["pagerank"].quantile(0.9):
        t.append("EXCHANGE_BYPASS")
    if sage_mag >= _sage_hi:
        t.append("GRAPH_ANOMALY")
    return t[:3] if t else ["ANOMALY"]


df["tags"] = [_tags(row, mag) for (_, row), mag in zip(df.iterrows(), _sage_norm)]

BTC_USD = 63847

# ============================================================
# LIVE GRAPH FORWARD PASS
# ============================================================
_GRAPH_X = GRAPH["x"]
_GRAPH_EDGE_INDEX = GRAPH["edge_index"]
_NODE_IDX = GRAPH["node_idx"]


@st.cache_resource(show_spinner=False)
def run_full_graph_forward():
    with torch.no_grad():
        emb, logits = sage_model(_GRAPH_X, _GRAPH_EDGE_INDEX)
        probs = F.softmax(logits, dim=1)[:, 1]
    return emb.cpu().numpy(), probs.cpu().numpy()


_ALL_SAGE_EMB, _ALL_SAGE_PROB = run_full_graph_forward()


def live_score_transaction(txid):
    if txid not in _NODE_IDX:
        return None
    idx = _NODE_IDX[txid]
    sage_vec = _ALL_SAGE_EMB[idx]
    sage_only_prob = float(_ALL_SAGE_PROB[idx])

    row = df.loc[df["txid"] == txid]
    if row.empty:
        return None
    row = row.iloc[0]

    feat_row = {c: row[c] for c in TABULAR_GRAPH_COLS}
    for i, c in enumerate(SAGE_COLS):
        feat_row[c] = sage_vec[i]

    feat_df = pd.DataFrame([feat_row])[FEATURES]
    hybrid_prob = float(model.predict_proba(feat_df)[:, 1])
    shap_vals = explainer(feat_df)

    return {
        "sage_only_prob": sage_only_prob,
        "hybrid_prob": hybrid_prob,
        "feat_df": feat_df,
        "shap_values": shap_vals,
        "row": row,
    }


SHAP_SAMPLE_SIZE = 400


@st.cache_resource(show_spinner=False)
def compute_global_shap():
    sample = df.sample(n=min(SHAP_SAMPLE_SIZE, len(df)), random_state=42)
    sample_X = sample[FEATURES]
    shap_out = explainer(sample_X)
    return sample, sample_X, shap_out


SHAP_SAMPLE_DF, SHAP_SAMPLE_X, GLOBAL_SHAP = compute_global_shap()
GLOBAL_MEAN_ABS_SHAP = (
    pd.Series(np.abs(GLOBAL_SHAP.values).mean(axis=0), index=FEATURES).sort_values(ascending=False)
)

# ============================================================
# REUSABLE UI COMPONENTS
# ============================================================
def kpi_card(label, value, sub=None, sub_class=""):
    sub_html = f'<div class="kpi-sub {sub_class}">{sub}</div>' if sub else ""
    st.markdown(
        f"""<div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                {sub_html}
            </div>""",
        unsafe_allow_html=True,
    )


def panel_start(title):
    st.markdown(f'<div class="panel"><div class="panel-title">{title}</div>', unsafe_allow_html=True)


def panel_end():
    st.markdown("</div>", unsafe_allow_html=True)


def risk_bar_html(pct, color):
    return (
        f'<div class="risk-track">'
        f'<div class="risk-fill" style="width:{pct}%; background:{color};"></div>'
        f"</div>"
    )


def render_pyvis(net: Network, height: int):
    """Render a pyvis network inline without writing a stray HTML file to disk."""
    html_str = net.generate_html(notebook=False)
    components.html(html_str, height=height, scrolling=False)


# ============================================================
# SIDEBAR — GLOBAL INTERACTIVE FILTERS
# ============================================================
with st.sidebar:
    st.markdown(
        '<div class="stitcher-title" style="font-size:18px;">ChainSentinel</div>'
        '<div class="stitcher-sub">Global Control Panel</div><br>',
        unsafe_allow_html=True,
    )

    search_query = st.text_input("🔍 Search wallet / txid / IP", placeholder="e.g. 1A1zP1... or CS-0004")

    sev_pick = st.multiselect(
        "Severity filter",
        ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        default=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
    )

    date_min, date_max = df["timestamp_chain"].min(), df["timestamp_chain"].max()
    date_range = st.slider(
        "Time window",
        min_value=date_min.to_pydatetime(),
        max_value=date_max.to_pydatetime(),
        value=(date_min.to_pydatetime(), date_max.to_pydatetime()),
    )

    min_conf = st.slider("Minimum AI confidence", 0.0, 1.0, 0.0, 0.05)

    st.markdown("---")
    auto_refresh = st.toggle("Live auto-refresh (30s)", value=False)
    st.caption("Filters apply across every tab in this session.")

# Apply the global filter set once, reused everywhere below
mask = (
    df["severity"].isin(sev_pick)
    & df["timestamp_chain"].between(date_range[0], date_range[1])
    & (df["risk_score"] >= min_conf)
)
if search_query:
    q = search_query.strip().lower()
    mask &= (
        df["wallet_out"].str.lower().str.contains(q, na=False)
        | df["wallet_in"].str.lower().str.contains(q, na=False)
        | df["txid"].str.lower().str.contains(q, na=False)
        | df["src_ip"].astype(str).str.lower().str.contains(q, na=False)
    )

df_f = df[mask].copy()
alert_df = df_f[df_f["risk_score"] >= 0.45].sort_values("risk_score", ascending=False)
n_critical = int((df_f["severity"] == "CRITICAL").sum())

if auto_refresh:
    st.markdown(
        '<meta http-equiv="refresh" content="30">',
        unsafe_allow_html=True,
    )

# ============================================================
# HEADER
# ============================================================
st.markdown(
    f"""<div class="stitcher-header">
            <div>
                <div class="stitcher-title">ChainSentinel Studio</div>
                <div class="stitcher-sub">GRAPH · GNN · SHAP — DEEP FORENSICS TARGET ARCHITECTURE</div>
            </div>
            <div class="stitcher-status-badge">HYBRID MODEL MODE</div>
        </div>""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""<div class="status-strip">
            <span><span class="dot-live"></span><b>Live</b> · GraphSAGE (Inductive) + XGBoost Ensemble</span>
            <span>Runtime: <b>Torch-PyG</b> core inference pipeline</span>
            <span>UTC App Time: <b>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}</b></span>
            <span>Environment: <b>SIH-OFFLINE-NODE</b></span>
            <span>Filtered rows: <b>{len(df_f):,}</b> / {len(df):,}</span>
        </div>""",
    unsafe_allow_html=True,
)

# ============================================================
# TABS
# ============================================================
tab_overview, tab_graph, tab_alerts, tab_ml, tab_gnn, tab_raw = st.tabs(
    [
        "🪐 Orchestration Overview",
        "🕸️ Interactive Network Map",
        "🚨 Triage Matrix",
        "🧠 Deep ML Explainability",
        "🔬 GNN Engineering Pipeline",
        "💾 Raw Ledger Feed",
    ]
)

# ------------------------------------------------------------
# TAB 1: OVERVIEW
# ------------------------------------------------------------
with tab_overview:
    if df_f.empty:
        st.info("No transactions match the current sidebar filters.")
    else:
        total_vol = df_f["total_input_amount"].sum()
        susp_vol = alert_df["total_input_amount"].sum()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Total Transactions", f"{len(df_f):,}", "Filtered observation window")
        with c2:
            kpi_card(
                "Flagged Entities",
                f"{len(alert_df):,}",
                f"{n_critical} critical anomalies",
                "up" if n_critical else "down",
            )
        with c3:
            kpi_card("Suspicious Volume", f"₿ {susp_vol:,.2f}", f"${susp_vol * BTC_USD:,.0f} USD equivalent")
        with c4:
            kpi_card("Total Flow Volume", f"₿ {total_vol:,.2f}", f"${total_vol * BTC_USD:,.0f} USD tracked")

        col_a, col_b = st.columns([1.6, 1])

        with col_a:
            panel_start("Network Transaction Flow Dynamics (Hourly Window)")
            daily = df_f.groupby(df_f["timestamp_chain"].dt.floor("h")).agg(total=("txid", "count")).reset_index()
            flagged_daily = (
                alert_df.groupby(alert_df["timestamp_chain"].dt.floor("h"))
                .agg(flagged=("txid", "count"))
                .reset_index()
            )
            merged = daily.merge(flagged_daily, on="timestamp_chain", how="left").fillna(0)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=merged["timestamp_chain"], y=merged["total"], name="Total Ledger Flow",
                    line=dict(color="#00F2FE", width=2), fill="tozeroy", fillcolor="rgba(0, 242, 254, 0.04)",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=merged["timestamp_chain"], y=merged["flagged"], name="AI Flagged Trajectories",
                    line=dict(color="#ef4444", width=2),
                )
            )
            fig.update_layout(
                template="plotly_dark", height=280, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
                font=dict(family="JetBrains Mono", size=10, color="#94a3b8"),
                xaxis=dict(rangeslider=dict(visible=True, thickness=0.06)),
            )
            st.plotly_chart(fig, width="stretch")
            panel_end()

        with col_b:
            panel_start("Threat Multi-Classification Vector")
            sev_counts = df_f["severity"].value_counts().reindex(["CRITICAL", "HIGH", "MEDIUM", "LOW"]).fillna(0)
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=sev_counts.index, values=sev_counts.values, hole=0.7,
                        marker=dict(colors=[SEV_COLOR[s] for s in sev_counts.index]), textinfo="none",
                    )
                ]
            )
            fig.update_layout(
                template="plotly_dark", height=200, margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
            )
            st.plotly_chart(fig, width="stretch")

            legend_html = "".join(
                f'<div style="display:flex; justify-content:space-between; padding:4px 0; font-size:12px;">'
                f'<span style="color:{SEV_COLOR[s]}">● {s}</span><span style="color:#f8fafc; font-weight:700;">{int(c)} Entities</span>'
                f"</div>"
                for s, c in sev_counts.items()
            )
            st.markdown(legend_html, unsafe_allow_html=True)
            panel_end()

        col_c, col_d = st.columns(2)

        with col_c:
            panel_start("Composite Systemic Risk Deviation Trend")
            risk_ts = df_f.groupby(df_f["timestamp_chain"].dt.floor("h"))["risk_score"].mean().reset_index()
            fig = go.Figure(
                go.Scatter(
                    x=risk_ts["timestamp_chain"], y=risk_ts["risk_score"], line=dict(color="#9B51E0", width=2),
                    fill="tozeroy", fillcolor="rgba(155, 81, 224, 0.08)",
                )
            )
            fig.update_layout(
                template="plotly_dark", height=230, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="JetBrains Mono", size=10, color="#94a3b8"),
            )
            st.plotly_chart(fig, width="stretch")
            panel_end()

        with col_d:
            panel_start("Geopolitical Network Origin Distribution (Top Autonomous Systems)")
            asn_counts = df_f["asn"].value_counts().head(6).sort_values()
            fig = go.Figure(
                go.Bar(
                    x=asn_counts.values, y=asn_counts.index, orientation="h",
                    marker=dict(color="rgba(0, 242, 254, 0.7)", line=dict(color="#00F2FE", width=1)),
                )
            )
            fig.update_layout(
                template="plotly_dark", height=230, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="JetBrains Mono", size=10, color="#94a3b8"),
            )
            st.plotly_chart(fig, width="stretch")
            panel_end()

        panel_start("Graph Cluster Core Risk Evaluation Matrix")
        cluster_agg = (
            df_f.groupby("cluster")
            .agg(wallets=("wallet_out", "nunique"), total_btc=("total_input_amount", "sum"), risk=("risk_score", "mean"))
            .sort_values("risk", ascending=False)
            .head(8)
            .reset_index()
        )
        hc = st.columns([1, 1, 1.2, 1, 1, 1.4])
        for c, t in zip(
            hc,
            ["TARGET CLUSTER", "CONNECTED WALLETS", "AGGREGATED VOLUME", "AI ESTIMATED RISK", "STATUS", "HYBRID DISPATCH RATIO"],
        ):
            c.markdown(f'<span class="panel-title">{t}</span>', unsafe_allow_html=True)

        for _, r in cluster_agg.iterrows():
            pct = r["risk"] * 100
            status = "EXPOSED" if pct >= 55 else "OPERATIONAL"
            badge_cls = "badge-flagged" if status == "EXPOSED" else "badge-clean"
            color = "#ef4444" if pct >= 70 else ("#f59e0b" if pct >= 45 else "#10b981")
            cols = st.columns([1, 1, 1.2, 1, 1, 1.4])
            cols[0].markdown(f'**{r["cluster"]}**')
            cols[1].markdown(f'{int(r["wallets"])}')
            cols[2].markdown(f'₿ {r["total_btc"]:.2f}')
            cols[3].markdown(f'{pct:.1f}%')
            cols[4].markdown(f'<span class="badge {badge_cls}">{status}</span>', unsafe_allow_html=True)
            cols[5].markdown(risk_bar_html(min(pct, 100), color), unsafe_allow_html=True)
        panel_end()

        st.download_button(
            "⬇ Export filtered overview data (.CSV)",
            df_f.to_csv(index=False),
            file_name="chainsentinel_overview_filtered.csv",
            mime="text/csv",
        )

# ------------------------------------------------------------
# TAB 2: INTERACTIVE NETWORK MAP
# ------------------------------------------------------------
with tab_graph:
    fcol1, fcol2, fcol3 = st.columns([2, 1, 1])
    with fcol1:
        min_risk = st.slider("Dynamic Risk Confidence Threshold", 0.0, 1.0, 0.45, 0.05, key="graph_risk_slider")
    with fcol2:
        max_nodes = st.number_input("Max nodes to render", min_value=10, max_value=200, value=60, step=10)
    with fcol3:
        physics_on = st.toggle("Physics simulation", value=True)

    graph_df = df_f[df_f["risk_score"] >= min_risk]

    gcol, sidecol = st.columns([2.2, 1])
    with gcol:
        panel_start(f"Subsurface Transaction Graph Execution View ({len(graph_df)} node links isolated)")
        if not graph_df.empty:
            net = Network(height="520px", width="100%", bgcolor="#070b12", font_color="#e2e8f0", directed=True)
            if physics_on:
                net.barnes_hut(gravity=-3500, spring_length=95)
            else:
                net.toggle_physics(False)

            plotted = graph_df.head(int(max_nodes))
            for _, row in plotted.iterrows():
                sev = row["severity"]
                color = SEV_COLOR[sev]
                size = 12 + row["risk_score"] * 22
                net.add_node(row["src_ip"], label=f"IP: {row['src_ip']}", color="#ef4444", shape="ellipse")
                net.add_node(row["wallet_in"], label=f"In: {row['wallet_in'][:6]}…", color="#4facfe", shape="box")
                net.add_node(row["wallet_out"], label=f"Out: {row['wallet_out'][:6]}…", color="#10b981", shape="box")
                net.add_node(row["txid"], label="TX", color=color, shape="dot", size=size,
                             title=f"Risk: {row['risk_score']*100:.1f}% · {sev}")
                net.add_edge(row["src_ip"], row["wallet_in"])
                net.add_edge(row["wallet_in"], row["txid"])
                net.add_edge(row["txid"], row["wallet_out"])

            render_pyvis(net, height=530)
        else:
            st.info("No network graphs intersect the current confidence bounds.")
        panel_end()

    with sidecol:
        panel_start("Target Risk Ledger Profile")
        if graph_df.empty:
            st.caption("No entities to profile at this threshold.")
        for _, r in graph_df.sort_values("risk_score", ascending=False).head(8).iterrows():
            badge_cls = SEV_BADGE[r["severity"]]
            st.markdown(
                f"""<div style="padding:10px 0; border-bottom:1px solid rgba(255,255,255,0.05);">
                        <div style="font-family:'JetBrains Mono'; font-size:12px; color:#f8fafc;">{r['wallet_out'][:14]}…</div>
                        <div style="font-size:11px; color:#64748b;">cluster mapping // {r['cluster']}</div>
                        <span class="badge {badge_cls}">{r['risk_score']*100:.0f}%</span>
                    </div>""",
                unsafe_allow_html=True,
            )
        panel_end()

# ------------------------------------------------------------
# TAB 3: TRIAGE MATRIX ALERTS
# ------------------------------------------------------------
with tab_alerts:
    sev_filter = st.radio(
        "Isolate Matrix Severity Tier", ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"],
        horizontal=True, label_visibility="collapsed",
    )
    view_df = alert_df if sev_filter == "ALL" else alert_df[alert_df["severity"] == sev_filter]

    panel_start(f"System Critical Triage Stream — ({len(view_df)} isolated execution contexts)")

    _alerts_shown = view_df.head(40)
    try:
        _alert_shap_vals = explainer(_alerts_shown[FEATURES]).values if not _alerts_shown.empty else None
    except Exception:
        _alert_shap_vals = None

    for i, (_, r) in enumerate(_alerts_shown.iterrows()):
        badge_cls = SEV_BADGE[r["severity"]]
        color = SEV_COLOR[r["severity"]]
        tags_html = "".join(f'<span class="badge badge-tag">{t}</span>' for t in r["tags"])
        unseen_html = '<span class="badge badge-unseen">HELD OUT INSTANCE</span>' if r.get("was_held_out") else ""

        header = (
            f'CS-{i+1:04d} · {r["severity"]} · ₿ {r["total_input_amount"]:.4f} '
            f'· {r["risk_score"]*100:.0f}% confidence · {r["wallet_out"][:10]}…'
        )
        with st.expander(header, expanded=False):
            top = st.columns([1, 6, 1.2])
            top[0].markdown(f'<span class="badge {badge_cls}">{r["severity"]}</span>', unsafe_allow_html=True)
            top[1].markdown(
                f"""<div style="font-family:'JetBrains Mono'; font-size:12px; color:#94a3b8;">
                        CONTEXT_ID: CS-{i+1:04d} · {r['timestamp_chain']} · {r['cluster']}
                    </div>
                    <div style="font-size:13px; color:#f8fafc; margin:4px 0;">
                        LEDGER_OUT: {r['wallet_out']} &nbsp;|&nbsp; ₿ {r['total_input_amount']:.4f}
                        (${r['total_input_amount']*BTC_USD:,.2f})
                    </div>
                    <div style="margin-top:6px;">{tags_html} {unseen_html}</div>""",
                unsafe_allow_html=True,
            )
            top[2].markdown(
                f"""<div style="text-align:right;">
                        <div class="kpi-value" style="font-size:22px;">{r["risk_score"]*100:.0f}%</div>
                        <div class="kpi-label">CONFIDENCE</div>
                        {risk_bar_html(r["risk_score"]*100, color)}
                    </div>""",
                unsafe_allow_html=True,
            )

            if _alert_shap_vals is not None and i < len(_alert_shap_vals):
                row_shap = pd.Series(_alert_shap_vals[i], index=FEATURES).sort_values(key=abs, ascending=False).head(5)
                chip_html = "".join(
                    f'<span style="font-size:11px; font-family:\'JetBrains Mono\'; '
                    f'color:{"#ef4444" if v > 0 else "#10b981"}; margin-right:14px;">'
                    f'{"▲" if v > 0 else "▼"} {f} ({v:+.2f})</span>'
                    for f, v in row_shap.items()
                )
                st.markdown(
                    f'<div style="margin-top:10px; font-size:11px; color:#64748b;">SHAP DECISION DRIVERS</div>'
                    f'<div style="margin-top:4px;">{chip_html}</div>',
                    unsafe_allow_html=True,
                )

    if len(view_df) > 40:
        st.caption(f"…Truncated {len(view_df) - 40} residual items from triage bounds. Refine filters to narrow the pool.")
    panel_end()

    st.download_button(
        "⬇ Export triage matrix (.CSV)",
        view_df.to_csv(index=False),
        file_name="chainsentinel_triage_matrix.csv",
        mime="text/csv",
    )

# ------------------------------------------------------------
# TAB 4: DEEP ML EXPLAINABILITY
# ------------------------------------------------------------
with tab_ml:
    c1, c2, c3 = st.columns([1.1, 1.4, 1])

    with c1:
        panel_start("Network Pipeline Profile")
        try:
            params = model.get_params()
        except Exception:
            params = {}
        n_trees = getattr(model, "n_estimators", params.get("n_estimators", "—"))
        max_depth = getattr(model, "max_depth", params.get("max_depth", "—"))
        rows = {
            "Core Model Variant": "GraphSAGE Inductive + XGBoost Ensemble",
            "Extracted Dimensions": len(FEATURES),
            "GraphSAGE Latent Vector": f"{len(SAGE_COLS)} Dimensions",
            "Structural/Tabular Topo": len(TABULAR_GRAPH_COLS),
            "XGBoost Sub-Estimators": n_trees,
            "Max Tree Structure Depth": max_depth,
            "Evaluated Context Vectors": f"{len(df_f):,} Instances",
            "Optimized Risk Bound": "0.45",
        }
        for k, v in rows.items():
            st.markdown(
                f'<div style="display:flex; justify-content:space-between; padding:6px 0; '
                f'border-bottom:1px solid rgba(255,255,255,0.04); font-size:12px;">'
                f'<span style="color:#94a3b8;">{k}</span><span style="color:#f8fafc; font-weight:700;">{v}</span></div>',
                unsafe_allow_html=True,
            )
        panel_end()

    with c2:
        panel_start("Structural Feature Vector Weighting (Gain Metrics)")
        try:
            importances = model.feature_importances_
            imp_series = pd.Series(importances, index=FEATURES).sort_values(ascending=True).tail(15)
            bar_colors = ["#9B51E0" if f in SAGE_COLS else "#00F2FE" for f in imp_series.index]
            fig = go.Figure(go.Bar(x=imp_series.values, y=imp_series.index, orientation="h", marker=dict(color=bar_colors)))
            fig.update_layout(
                template="plotly_dark", height=320, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="JetBrains Mono", size=10, color="#94a3b8"),
            )
            st.plotly_chart(fig, width="stretch")
            st.markdown("🟣 GraphSAGE Inductive Space &nbsp;&nbsp; 🔵 Structural Ledger Heuristics", unsafe_allow_html=True)
        except Exception:
            st.info("Feature distribution map offline.")
        panel_end()

    with c3:
        panel_start("Inference Quality Verification Matrix")
        if "label" in df_f.columns and not df_f.empty:
            tp = int(((df_f["label"] == 1) & (df_f["risk_score"] >= 0.45)).sum())
            fp = int(((df_f["label"] == 0) & (df_f["risk_score"] >= 0.45)).sum())
            fn = int(((df_f["label"] == 1) & (df_f["risk_score"] < 0.45)).sum())
            tn = int(((df_f["label"] == 0) & (df_f["risk_score"] < 0.45)).sum())
            m1, m2 = st.columns(2)
            m1.markdown(f'<div class="kpi-label">TRUE POS</div><div class="kpi-value" style="font-size:22px;">{tp}</div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="kpi-label">FALSE POS</div><div class="kpi-value" style="font-size:22px;">{fp}</div>', unsafe_allow_html=True)
            m3, m4 = st.columns(2)
            m3.markdown(f'<div class="kpi-label">FALSE NEG</div><div class="kpi-value" style="font-size:22px;">{fn}</div>', unsafe_allow_html=True)
            m4.markdown(f'<div class="kpi-label">TRUE NEG</div><div class="kpi-value" style="font-size:22px;">{tn}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="kpi-label">FLAGGED MATRIX VOLUME</div>'
                f'<div class="kpi-value" style="font-size:22px;">{len(alert_df)} / {len(df_f)}</div>',
                unsafe_allow_html=True,
            )
        panel_end()

    shc1, shc2 = st.columns([1, 1.5])
    with shc1:
        panel_start(f"Global SHAP Vector Importance Map (n={len(SHAP_SAMPLE_DF)})")
        top_shap = GLOBAL_MEAN_ABS_SHAP.head(15).sort_values(ascending=True)
        bar_colors = ["#9B51E0" if f in SAGE_COLS else "#00F2FE" for f in top_shap.index]
        fig = go.Figure(go.Bar(x=top_shap.values, y=top_shap.index, orientation="h", marker=dict(color=bar_colors)))
        fig.update_layout(
            template="plotly_dark", height=360, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono", size=10, color="#94a3b8"),
        )
        st.plotly_chart(fig, width="stretch")
        panel_end()

    with shc2:
        panel_start("SHAP Summary Density Feature Scatter Matrix")
        top_feats = GLOBAL_MEAN_ABS_SHAP.head(10).index.tolist()
        shap_vals_arr = GLOBAL_SHAP.values
        rng = np.random.RandomState(42)
        fig = go.Figure()
        for i, feat in enumerate(top_feats):
            fi = FEATURES.index(feat)
            vals = shap_vals_arr[:, fi]
            raw = SHAP_SAMPLE_X[feat].values.astype(float)
            rmin, rmax = np.percentile(raw, 1), np.percentile(raw, 99)
            norm = np.clip((raw - rmin) / (rmax - rmin), 0, 1) if rmax > rmin else np.zeros_like(raw)
            jitter = (rng.rand(len(vals)) - 0.5) * 0.75
            fig.add_trace(
                go.Scatter(
                    x=vals, y=[len(top_feats) - 1 - i + j for j in jitter], mode="markers",
                    marker=dict(
                        size=4.5, color=norm, colorscale="IceFire", opacity=0.8, showscale=(i == 0),
                        colorbar=dict(title="Feature Value", tickvals=[0, 1], ticktext=["Low", "High"], len=0.75)
                        if i == 0 else None,
                    ),
                    name=feat, showlegend=False, hovertemplate=f"{feat}<br>SHAP: %{{x:.4f}}<extra></extra>",
                )
            )
        fig.add_vline(x=0, line_color="rgba(255,255,255,0.15)", line_width=1, line_dash="dot")
        fig.update_layout(
            template="plotly_dark", height=360, margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(tickmode="array", tickvals=list(range(len(top_feats))), ticktext=list(reversed(top_feats)), automargin=True),
            xaxis_title="SHAP Value Matrix Impact Estimation", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono", size=10, color="#94a3b8"),
        )
        st.plotly_chart(fig, width="stretch")
        panel_end()

# ------------------------------------------------------------
# TAB 5: GNN SUBSURFACE STRUCTURAL ENGINEERING
# ------------------------------------------------------------
with tab_gnn:
    panel_start("Hybrid Layer Processing Architecture Topology")
    stages = [
        ("Subsurface Graph", "Source Context", "Wallets / Transactions", "#00F2FE"),
        ("GraphSAGE L1", "SAGEConv Block", f"{META['hidden']}-d Space", "#9B51E0"),
        ("GraphSAGE L2", "SAGEConv Feature", f"{META['emb_dim']}-d Encoding", "#9B51E0"),
        ("Concatenation", "Fusion Space", "Hybrid Features Vector", "#4facfe"),
        ("XGBoost Layer", "Ensemble Matrix", "Tree Gradient Evaluation", "#f59e0b"),
        ("Risk Output", "Probability Bounds", "P(Illicit Vector)", "#ef4444"),
    ]
    scols = st.columns(len(stages) * 2 - 1)
    for i, (title, sub, note, color) in enumerate(stages):
        scols[i * 2].markdown(
            f"""<div style="text-align:center;">
                    <div style="font-size:13px; font-weight:800; color:{color};">{title}</div>
                    <div style="font-size:11px; color:#94a3b8; margin-top:2px;">{sub}</div>
                    <div style="font-size:10px; color:#64748b; font-family:'JetBrains Mono'; margin-top:2px;">{note}</div>
                </div>""",
            unsafe_allow_html=True,
        )
        if i < len(stages) - 1:
            scols[i * 2 + 1].markdown(
                '<div style="text-align:center; color:#334155; font-size:20px; padding-top:8px;">→</div>',
                unsafe_allow_html=True,
            )
    panel_end()

    panel_start("🧠 Live Inductive Network Node Evaluation Engine")
    scope = st.radio(
        "Isolate Target Evaluation Pool",
        ["Flagged alerts", "Held out of training (closest stand-in for 'unseen')"],
        horizontal=True, label_visibility="collapsed", key="scope_radio_gnn",
    )
    pool = (
        df_f[df_f["was_held_out"]]
        if (scope.startswith("Held out") and "was_held_out" in df_f.columns and not df_f[df_f["was_held_out"]].empty)
        else alert_df
    )

    if not pool.empty:
        selected_tx = st.selectbox("Isolate ledger transaction address:", pool["txid"].values, key="live_tx_select_gnn")
        res = live_score_transaction(selected_tx)

        if res is None:
            st.warning("Transaction execution parameters missing from isolated snapshot.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.markdown(
                f'<div class="kpi-label">GraphSAGE Latent Score</div>'
                f'<div class="kpi-value" style="font-size:24px;">{res["sage_only_prob"]*100:.1f}%</div>',
                unsafe_allow_html=True,
            )
            m2.markdown(
                f'<div class="kpi-label">Hybrid Pipeline Risk Engine</div>'
                f'<div class="kpi-value" style="font-size:24px;">{res["hybrid_prob"]*100:.1f}%</div>',
                unsafe_allow_html=True,
            )
            was_unseen = bool(res["row"].get("was_held_out", False))
            m3.markdown(
                f'<div class="kpi-label">Inductive Blind Inference?</div>'
                f'<div class="kpi-value" style="font-size:24px; color:{"#10b981" if was_unseen else "#64748b"};">'
                f'{"TRUE" if was_unseen else "FALSE"}</div>',
                unsafe_allow_html=True,
            )

            ecol, gcol = st.columns([1, 1.1])
            with ecol:
                st.markdown('<div class="panel-title">EXPLANATION MATRIX (LOCAL SHAP WEIGHTS)</div>', unsafe_allow_html=True)
                shap_vals = res["shap_values"].values
                base_val = res["shap_values"].base_values
                shap_df = (
                    pd.DataFrame(
                        {
                            "Feature": FEATURES,
                            "Value": [res["feat_df"].iloc[0][f] for f in FEATURES],
                            "SHAP Impact": shap_vals[0] if np.ndim(shap_vals) > 1 else shap_vals,
                        }
                    )
                    .sort_values(by="SHAP Impact", key=abs, ascending=False)
                    .head(10)
                )

                def highlight_impact(val):
                    return f'color: {"#ef4444" if val > 0 else "#10b981"}; font-weight: bold; font-family: "JetBrains Mono";'

                base_display = float(np.ravel(base_val)[0]) if np.ndim(base_val) else float(base_val)
                st.markdown(f"Model Baseline: **{base_display*100:.2f}%**")
                st.dataframe(shap_df.style.map(highlight_impact, subset=["SHAP Impact"]), width="stretch", height=280)

            with gcol:
                st.markdown('<div class="panel-title">ISOLATED TRANSACTION LOCAL LINK VIEW</div>', unsafe_allow_html=True)
                row = res["row"]
                net = Network(height="330px", width="100%", bgcolor="#070b12", font_color="#e2e8f0", directed=True)
                net.add_node(row["src_ip"], label=f"IP: {row['src_ip']}", color="#ef4444", shape="ellipse")
                net.add_node(row["wallet_in"], label=f"Inbound: {row['wallet_in'][:8]}…", color="#4facfe", shape="box")
                net.add_node(row["wallet_out"], label=f"Outbound: {row['wallet_out'][:8]}…", color="#10b981", shape="box")
                net.add_node(selected_tx, label="Target Tx", color="#9B51E0", shape="dot")
                net.add_edge(row["src_ip"], row["wallet_in"])
                net.add_edge(row["wallet_in"], selected_tx)
                net.add_edge(selected_tx, row["wallet_out"])
                render_pyvis(net, height=340)
    else:
        st.info("No transactions available in the selected evaluation pool under current filters.")
    panel_end()

# ------------------------------------------------------------
# TAB 6: RAW LEDGER FEED
# ------------------------------------------------------------
with tab_raw:
    panel_start("System Combined Feature Matrix Database Log")
    display_cols = [c for c in df_f.columns if not c.startswith("sage_emb_")]
    st.dataframe(df_f[display_cols].sort_values("timestamp_chain", ascending=False), width="stretch", height=560)
    st.caption(
        "Latent GraphSAGE node-embedding vectors are omitted from this interface render to balance thread "
        "performance. Use the exporter to retrieve full datasets."
    )
    st.download_button(
        "⬇ Export Complete System Data Frame (.CSV)",
        df_f.to_csv(index=False),
        file_name="chainsentinel_complete_export.csv",
        mime="text/csv",
    )
    panel_end()