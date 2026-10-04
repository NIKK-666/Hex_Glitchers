"""ChainSentinel inference engine (no Streamlit).

Loads the same offline bundle the Streamlit dashboard uses (data2/) and exposes
plain-Python methods that the FastAPI layer in main.py turns into JSON.
"""
from __future__ import annotations

import os
import pickle
import warnings
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

DATA_DIR = os.environ.get("CS_DATA_DIR", "data2")
DEVICE = torch.device("cpu")
BTC_USD = 63847

TYPOLOGY = {
    "TOR_RELAY": "Darknet / Tor relay",
    "STRUCTURING": "Layering / structuring",
    "HIGH_VELOCITY": "Rapid pass-through",
    "ROUND_AMOUNT": "Round-amount transfer",
    "MIXING_SERVICE": "Mixer / tumbler",
    "EXCHANGE_BYPASS": "Exchange cash-out",
    "GRAPH_ANOMALY": "Graph anomaly",
    "ANOMALY": "Unclassified anomaly",
}
SIGNAL_TEXT = {
    "TOR_RELAY": "Source IP is a known Tor / threat relay",
    "STRUCTURING": "Fee in top 15% (structuring)",
    "HIGH_VELOCITY": "Latency in top 15% (high velocity)",
    "ROUND_AMOUNT": "Round-number input amount",
    "MIXING_SERVICE": "Script type typical of mixing services",
    "EXCHANGE_BYPASS": "PageRank in top 10% (hub behaviour)",
    "GRAPH_ANOMALY": "GraphSAGE embedding magnitude in top 10%",
    "ANOMALY": "Model score above threshold",
}


class GraphSAGEEncoder(torch.nn.Module):
    """Identical to the class in the Streamlit dashboard (weights must match)."""

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


def severity(s: float) -> str:
    if s >= 0.85:
        return "CRITICAL"
    if s >= 0.65:
        return "HIGH"
    if s >= 0.45:
        return "MEDIUM"
    return "LOW"


def short(addr: Any, n: int = 6) -> str:
    a = str(addr)
    return a if len(a) <= 2 * n + 1 else f"{a[:n]}…{a[-4:]}"


class Engine:
    def __init__(self) -> None:
        self._load()
        self._prepare()
        self.status: dict[str, str] = {}  # alert_id -> analyst status (in-memory)

    # ------------------------------------------------------------------ load
    def _load(self) -> None:
        p = lambda f: os.path.join(DATA_DIR, f)  # noqa: E731
        with open(p("xgboost_forensics_model.pkl"), "rb") as f:
            self.model = pickle.load(f)
        with open(p("shap_explainer.pkl"), "rb") as f:
            self.explainer = pickle.load(f)
        with open(p("pipeline_metadata.pkl"), "rb") as f:
            self.meta = pickle.load(f)
        with open(p("graph_snapshot.pkl"), "rb") as f:
            self.graph = pickle.load(f)
        self.df = pd.read_pickle(p("scored_data.pkl")).copy()

        self.sage = GraphSAGEEncoder(self.meta["in_dim"], self.meta["hidden"], self.meta["emb_dim"])
        self.sage.load_state_dict(torch.load(p("graphsage_encoder.pt"), map_location=DEVICE))
        self.sage.eval()

        self.features: list[str] = list(self.meta["features"])
        self.sage_cols: list[str] = list(self.meta["sage_cols"])
        self.tab_cols = [c for c in self.features if c not in self.sage_cols]

        self.geoip = None
        mmdb = p("GeoLite2-City.mmdb")
        if os.path.exists(mmdb):
            try:
                import geoip2.database

                self.geoip = geoip2.database.Reader(mmdb)
            except Exception:
                self.geoip = None

    # --------------------------------------------------------------- prepare
    def _prepare(self) -> None:
        df = self.df
        df["timestamp_chain"] = pd.to_datetime(df["timestamp_chain"])
        if "risk_score" not in df.columns:
            df["risk_score"] = self.model.predict_proba(df[self.features])[:, 1]
        df["severity"] = df["risk_score"].apply(severity)
        df["cluster"] = "C-" + pd.Series(
            pd.factorize(df["wallet_out"])[0] % 14 + 1, index=df.index
        ).astype(str).str.zfill(2)

        # vectorised version of the dashboard's _tags()
        sage_norm = np.linalg.norm(df[self.sage_cols].values, axis=1)
        script_mode = df["script_encoded"].mode().iloc[0]
        threat = (df["src_is_threat"].fillna(0).astype(int) == 1) if "src_is_threat" in df else pd.Series(False, index=df.index)
        masks = [
            ("TOR_RELAY", threat.values),
            ("STRUCTURING", (df["fee"] >= df["fee"].quantile(0.85)).values),
            ("HIGH_VELOCITY", (df["latency_seconds"] >= df["latency_seconds"].quantile(0.85)).values),
            ("ROUND_AMOUNT", (np.mod(df["total_input_amount"].astype(float), 1) == 0)),
            ("MIXING_SERVICE", (df["script_encoded"] == script_mode).values),
            ("EXCHANGE_BYPASS", (df["pagerank"] >= df["pagerank"].quantile(0.9)).values),
            ("GRAPH_ANOMALY", sage_norm >= np.quantile(sage_norm, 0.9)),
        ]
        names = [n for n, _ in masks]
        M = np.column_stack([m for _, m in masks])
        df["tags"] = [([names[i] for i in np.flatnonzero(r)][:3] or ["ANOMALY"]) for r in M]

        # GraphSAGE forward pass (once)
        with torch.no_grad():
            emb, logits = self.sage(self.graph["x"], self.graph["edge_index"])
            self.sage_emb = emb.cpu().numpy()
            self.sage_prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        self.node_idx = self.graph["node_idx"]

        # geo lookup per unique IP
        self.geo = self._geolocate(df["src_ip"].dropna().unique().tolist()) if "src_ip" in df else {}

        # wallet -> row positions, for graph / trace
        self.by_in: dict[str, list[int]] = defaultdict(list)
        self.by_out: dict[str, list[int]] = defaultdict(list)
        for pos, (wi, wo) in enumerate(zip(df["wallet_in"].values, df["wallet_out"].values)):
            self.by_in[wi].append(pos)
            self.by_out[wo].append(pos)

        # global SHAP on a sample (same as dashboard)
        sample = df.sample(n=min(400, len(df)), random_state=42)
        sv = self.explainer(sample[self.features])
        vals = self._shap_matrix(sv)
        self.global_importance = (
            pd.Series(np.abs(vals).mean(axis=0), index=self.features).sort_values(ascending=False)
        )
        self.feat_range = {
            c: (float(df[c].quantile(0.01)), float(df[c].quantile(0.99))) for c in self.tab_cols
        }
        for c in self.sage_cols:
            self.feat_range[c] = (float(np.quantile(self.sage_emb[:, self.sage_cols.index(c)], 0.01)),
                                  float(np.quantile(self.sage_emb[:, self.sage_cols.index(c)], 0.99)))

        # extra lookups for the real graph explorer
        self.by_ip: dict[str, list[int]] = defaultdict(list)
        if "src_ip" in df:
            for pos, ip in enumerate(df["src_ip"].values):
                if pd.notna(ip):
                    self.by_ip[str(ip)].append(pos)
        self._ts = df["timestamp_chain"].values
        self._amt = df["total_input_amount"].astype(float).values
        self._risk = df["risk_score"].values

        # network correlation source (correlated_data.pkl if usable, else scored_data.pkl)
        self._build_corr()

    def _geolocate(self, ips: list[str]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for ip in ips:
            rec = {"country": None, "code": None, "lat": None, "lon": None}
            if self.geoip is not None:
                try:
                    r = self.geoip.city(ip)
                    rec = {"country": r.country.name, "code": r.country.iso_code,
                           "lat": r.location.latitude, "lon": r.location.longitude}
                except Exception:
                    pass
            out[ip] = rec
        return out

    @staticmethod
    def _shap_matrix(sv) -> np.ndarray:
        v = np.asarray(sv.values)
        return v[:, :, 1] if v.ndim == 3 else v

    # --------------------------------------------------------------- scoring
    def _feature_row(self, txid: str, overrides: dict[str, float] | None = None) -> pd.DataFrame | None:
        if txid not in self.node_idx:
            return None
        hit = self.df.loc[self.df["txid"] == txid]
        if hit.empty:
            return None
        row = hit.iloc[0]
        feat = {c: row[c] for c in self.tab_cols}
        vec = self.sage_emb[self.node_idx[txid]]
        for i, c in enumerate(self.sage_cols):
            feat[c] = vec[i]
        for k, v in (overrides or {}).items():
            if k in feat:
                feat[k] = v
        return pd.DataFrame([feat])[self.features]

    def explain(self, txid: str, top: int = 8) -> dict | None:
        fdf = self._feature_row(txid)
        if fdf is None:
            return None
        hybrid = float(self.model.predict_proba(fdf)[:, 1][0])
        sv = self.explainer(fdf)
        vals = self._shap_matrix(sv)[0]
        base = np.asarray(sv.base_values).ravel()
        base_val = float(base[-1]) if base.size else 0.0
        order = np.argsort(-np.abs(vals))[:top]
        feats = []
        for i in order:
            name = self.features[i]
            lo, hi = self.feat_range.get(name, (0.0, 1.0))
            feats.append({"feature": name, "value": float(fdf.iloc[0][name]), "shap": float(vals[i]),
                          "min": lo, "max": hi if hi > lo else lo + 1.0})
        return {
            "txid": txid,
            "hybrid_prob": hybrid,
            "sage_only_prob": float(self.sage_prob[self.node_idx[txid]]),
            "base_value": base_val,
            "features": feats,
        }

    def what_if(self, txid: str, overrides: dict[str, float]) -> dict | None:
        fdf = self._feature_row(txid, overrides)
        if fdf is None:
            return None
        p = float(self.model.predict_proba(fdf)[:, 1][0])
        return {"txid": txid, "risk_score": round(p * 100), "severity": severity(p)}

    # --------------------------------------------------------------- alerts
    def _alert(self, r: pd.Series, rank_id: int) -> dict:
        txid = str(r["txid"])
        aid = f"ALT-{rank_id:03d}"
        tags = list(r["tags"])
        geo = self.geo.get(r.get("src_ip"), {}) if "src_ip" in r else {}
        ts = r["timestamp_chain"]
        fmt = lambda t: t.strftime("%d %b %H:%M")  # noqa: E731
        score = float(r["risk_score"])
        conf = int(round(max(score, 1 - score) * 100))
        return {
            "id": aid,
            "txid": txid,
            "entity": r["cluster"],
            "wallet_id": short(r["wallet_out"]),
            "wallet_full": str(r["wallet_out"]),
            "risk_score": int(round(score * 100)),
            "confidence": conf,
            "typology": TYPOLOGY[tags[0]],
            "contributing_signals": [SIGNAL_TEXT[t] for t in tags],
            "matched_pattern": TYPOLOGY[tags[0]],
            "shap_features": [],  # filled by /api/alerts/{id}
            "propagation_path": [short(r["wallet_in"]), r["cluster"], short(r["wallet_out"])],
            "country": geo.get("code") or "—",
            "asn": str(r.get("asn", "—")),
            "status": self.status.get(aid, "New"),
            "firstSeen": fmt(ts),
            "lastSeen": fmt(ts),
        }

    def alerts(self, min_risk: float = 0.45, limit: int = 100) -> dict:
        top = self.df[self.df["risk_score"] >= min_risk].sort_values("risk_score", ascending=False)
        items = [self._alert(r, i + 1) for i, (_, r) in enumerate(top.head(limit).iterrows())]
        return {"total": int(len(top)), "open": sum(1 for a in items if a["status"] in ("New", "Investigating")), "items": items}

    def alert_by_txid(self, txid: str) -> dict | None:
        hit = self.df.loc[self.df["txid"] == txid]
        if hit.empty:
            return None
        r = hit.iloc[0]
        rank = int((self.df["risk_score"] > r["risk_score"]).sum()) + 1
        a = self._alert(r, rank)
        ex = self.explain(txid, top=3)
        if ex:
            a["shap_features"] = [{"feature": f["feature"], "value": f"{f['shap']:+.2f}"} for f in ex["features"]]
        return a

    def set_status(self, alert_id: str, status: str) -> None:
        self.status[alert_id] = status

    # -------------------------------------------------------------- overview
    def overview(self) -> dict:
        df = self.df
        flagged = df[df["risk_score"] >= 0.65]
        ts = df["timestamp_chain"]
        bins = pd.cut(ts, 7, labels=False)

        def per_bin(fn) -> list[float]:
            return [round(float(fn(df[bins == b])), 2) if (bins == b).any() else 0.0 for b in range(7)]

        def delta(series: list[float]) -> str:
            a, b = series[-2], series[-1]
            return "no prior bin" if not a else f"{(b - a) / a * 100:+.1f}% vs prior"

        tx_s = per_bin(len)
        net_s = per_bin(lambda d: d["src_ip"].nunique() if "src_ip" in d else 0)
        flag_s = per_bin(lambda d: (d["risk_score"] >= 0.65).sum())
        crit_s = per_bin(lambda d: (d["risk_score"] >= 0.85).sum())
        risk_s = per_bin(lambda d: d["risk_score"].mean() * 100)
        mix_s = per_bin(lambda d: d["tags"].apply(lambda t: "MIXING_SERVICE" in t).sum())

        links = df[["src_ip", "wallet_in"]].drop_duplicates().shape[0] if "src_ip" in df else 0
        mix_hits = int(df["tags"].apply(lambda t: "MIXING_SERVICE" in t).sum())
        cash = flagged[flagged["tags"].apply(lambda t: "EXCHANGE_BYPASS" in t)]["total_input_amount"].sum()

        def k(v, sfx="", d="", pts=None, muted=False):
            return {"value": v, "suffix": sfx, "delta": d, "points": pts or [1] * 7, "muted": muted}

        def big(n):  # 12345 -> ("12.3","K")
            for lim, s in ((1e6, "M"), (1e3, "K")):
                if n >= lim:
                    return f"{n / lim:.1f}", s
            return f"{int(n)}", ""

        v, s = big(len(df)); kt = k(v, s, delta(tx_s), tx_s)
        v, s = big(df["src_ip"].notna().sum() if "src_ip" in df else 0); kn = k(v, s, delta(net_s), net_s)
        kpis = {
            "tx": kt, "network": kn,
            "flagged": k(f"{flagged['wallet_out'].nunique()}", "", f"{len(flagged)} tx ≥ 0.65", flag_s),
            "critical": k(f"{int((df['risk_score'] >= 0.85).sum()):02d}", "", "risk ≥ 0.85", crit_s),
            "links": k(*big(links)[:1], big(links)[1], "unique IP↔wallet pairs", net_s),
            "volume": k(f"{flagged['total_input_amount'].sum():.1f}", "BTC", f"≈ ${flagged['total_input_amount'].sum() * BTC_USD:,.0f}", flag_s),
            "clusters": k(f"{df['cluster'].nunique()}", "", "dashboard clustering", [1] * 7),
            "avg_risk": k(f"{df['risk_score'].mean() * 100:.1f}", "", delta(risk_s), risk_s),
            "confidence": k(f"{float(np.mean(np.maximum(df['risk_score'], 1 - df['risk_score'])) * 100):.1f}", "%", "mean max(p, 1-p)", risk_s),
            "mixer": k(f"{mix_hits}", "", "script-type heuristic", mix_s),
            "cashout": k(f"{cash:.1f}", "BTC", "flagged hub-wallet volume", flag_s),
            "health": k("Online", "", "engine loaded", [1] * 7),
        }

        # hourly telemetry (2h buckets)
        g = df.assign(h=(ts.dt.hour // 2) * 2).groupby("h")
        telemetry = [{
            "time": f"{int(h):02d}:00",
            "tx": int(len(d)),
            "network": int(d["src_ip"].nunique()) if "src_ip" in d else 0,
            "flagged": int((d["risk_score"] >= 0.65).sum()),
            "score": round(float(d["risk_score"].mean() * 100)),
        } for h, d in g]

        sev = df["severity"].value_counts()
        palette = {"CRITICAL": "#f472d0", "HIGH": "#ef4f79", "MEDIUM": "#ffe23a", "LOW": "#6bff6b"}
        risk_dist = [{"name": n.title(), "value": int(sev.get(n, 0)), "color": palette[n]} for n in ("CRITICAL", "HIGH", "MEDIUM", "LOW")]

        tag_counts = pd.Series([t for ts_ in flagged["tags"] for t in ts_]).value_counts().head(5)
        pal = ["#f472d0", "#2fb8ff", "#8b5cf6", "#ffe23a", "#6b7280"]
        typologies = [{"name": TYPOLOGY[n].split(" / ")[0], "value": int(c), "color": pal[i]} for i, (n, c) in enumerate(tag_counts.items())]

        geo = self.geo_summary()
        asn = df["asn"].value_counts().head(1) if "asn" in df else pd.Series(dtype=int)
        return {
            "kpis": kpis,
            "telemetry": telemetry,
            "risk_distribution": risk_dist,
            "typologies": typologies,
            "geo": [{"country": c["code"] or c["name"], "value": c["alerts"]} for c in geo[:5]],
            "top_asn": str(asn.index[0]) if len(asn) else "—",
            "status": {
                "tx": int(len(df)),
                "wallets": int(pd.unique(df[["wallet_in", "wallet_out"]].values.ravel()).shape[0]),
                "alerts": int((df["risk_score"] >= 0.45).sum()),
                "snapshot": ts.max().strftime("%Y-%m-%d"),
                "high_or_critical": int((df["risk_score"] >= 0.65).sum()),
            },
        }

    # ------------------------------------------------------------------- geo
    def geo_summary(self) -> list[dict]:
        if "src_ip" not in self.df:
            return []
        d = self.df.assign(
            country=self.df["src_ip"].map(lambda i: self.geo.get(i, {}).get("country")),
            code=self.df["src_ip"].map(lambda i: self.geo.get(i, {}).get("code")),
        ).dropna(subset=["country"])
        out = []
        for (country, code), g in d.groupby(["country", "code"]):
            out.append({
                "name": country, "code": code,
                "alerts": int((g["risk_score"] >= 0.45).sum()),
                "volume": f"{g['total_input_amount'].sum():.1f} BTC",
                "score": int(round(g["risk_score"].mean() * 100)),
            })
        return sorted(out, key=lambda x: -x["score"])

    # -------------------------------------------------------------- clusters
    def clusters(self) -> list[dict]:
        df = self.df
        out = []
        for cid, g in df.groupby("cluster"):
            top_tag = pd.Series([t for ts_ in g["tags"] for t in ts_]).value_counts().index[0]
            codes = g["src_ip"].map(lambda i: self.geo.get(i, {}).get("code")).dropna() if "src_ip" in g else pd.Series(dtype=str)
            out.append({
                "id": cid,
                "wallet_count": int(g["wallet_out"].nunique()),
                "total_volume": round(float(g["total_input_amount"].sum()), 1),
                "method": "Wallet-out grouping",
                "risk": int(round(g["risk_score"].mean() * 100)),
                "typology": TYPOLOGY[top_tag],
                "country": codes.mode().iloc[0] if len(codes) else "—",
                "ips": int(g["src_ip"].nunique()) if "src_ip" in g else 0,
            })
        return sorted(out, key=lambda x: -x["risk"])

    # ----------------------------------------------------------------- model
    def model_card(self) -> dict:
        df = self.df
        metrics = None
        if "label" in df.columns:
            ev = df[df["was_held_out"].astype(bool)] if "was_held_out" in df.columns and df["was_held_out"].astype(bool).any() else df
            tp = int(((ev["label"] == 1) & (ev["risk_score"] >= 0.45)).sum())
            fp = int(((ev["label"] == 0) & (ev["risk_score"] >= 0.45)).sum())
            fn = int(((ev["label"] == 1) & (ev["risk_score"] < 0.45)).sum())
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            auc = None
            try:
                from sklearn.metrics import roc_auc_score

                if ev["label"].nunique() == 2:
                    auc = float(roc_auc_score(ev["label"], ev["risk_score"]))
            except Exception:
                pass
            metrics = {"precision": prec, "recall": rec, "f1": f1, "auc": auc,
                       "n_eval": int(len(ev)), "held_out_only": bool(len(ev) != len(df))}
        return {
            "model": "XGBoost on tabular + GraphSAGE embeddings",
            "n_features": len(self.features),
            "sage": {"hidden": int(self.meta["hidden"]), "emb_dim": int(self.meta["emb_dim"])},
            "threshold": 0.45,
            "metrics": metrics,
            "importance": [{"feature": k, "value": float(v)} for k, v in self.global_importance.head(10).items()],
        }

    # ------------------------------------------------------- graph / tracing
    def _wallet_risk(self, w: str) -> int:
        pos = self.by_in.get(w, []) + self.by_out.get(w, [])
        return int(round(float(self.df["risk_score"].values[pos].max()) * 100)) if pos else 0

    def top_wallet(self) -> str:
        return str(self.df.sort_values("risk_score", ascending=False).iloc[0]["wallet_out"])

    def _node_props(self, kind: str, key: str) -> list[list[str]]:
        fmt = lambda t: str(pd.Timestamp(t).strftime("%d %b %Y %H:%M"))  # noqa: E731
        if kind == "tx":
            r = self.df.loc[self.df["txid"] == key].iloc[0]
            return [["timestamp", fmt(r["timestamp_chain"])], ["amount", f"{float(r['total_input_amount']):.4f} BTC"],
                    ["fee", f"{float(r['fee']):.6g}"]]
        if kind == "ip":
            pos = self.by_ip.get(key, [])
            if not pos:
                return []
            g = self.geo.get(key, {})
            return [["first seen", fmt(self._ts[pos].min())], ["last seen", fmt(self._ts[pos].max())],
                    ["tx / country", f"{len(pos)} / {g.get('code') or '—'}"]]
        sent, recv = self.by_in.get(key, []), self.by_out.get(key, [])
        allp = sent + recv
        if not allp:
            return []
        return [["first seen", fmt(self._ts[allp].min())], ["last seen", fmt(self._ts[allp].max())],
                ["out / in volume", f"{self._amt[sent].sum():.2f} / {self._amt[recv].sum():.2f} BTC"]]

    def subgraph(self, q: str | None = None, min_risk: float = 0.45, max_rows: int = 15) -> dict:
        """Same structure as the Streamlit map: IP -> wallet_in -> TX -> wallet_out,
        laid out in four columns. Rows are the highest-risk transactions that pass the filter."""
        df = self.df
        ql = (q or "").strip().lower()
        if ql:  # lookups ignore the risk slider so an analyst can find any wallet / IP / txid
            m = (df["wallet_in"].astype(str).str.lower().str.contains(ql, regex=False)
                 | df["wallet_out"].astype(str).str.lower().str.contains(ql, regex=False)
                 | df["txid"].astype(str).str.lower().str.contains(ql, regex=False))
            if "src_ip" in df:
                m |= df["src_ip"].astype(str).str.lower().str.contains(ql, regex=False)
            sub = df[m]
        else:
            sub = df[df["risk_score"] >= min_risk]
        sub = sub.sort_values("risk_score", ascending=False).head(max_rows)

        cols: dict[str, dict[str, float]] = {"ip": {}, "win": {}, "tx": {}, "wout": {}}
        placed: set[str] = set()
        edges: list[list[str]] = []
        seen_edges: set[tuple[str, str]] = set()

        def add(col: str, key: str, risk: float) -> str:
            if key not in placed:
                placed.add(key)
                cols[col][key] = risk
            else:
                for c in cols.values():
                    if key in c:
                        c[key] = max(c[key], risk)
            return key

        def edge(a: str, b: str) -> None:
            if (a, b) not in seen_edges:
                seen_edges.add((a, b))
                edges.append([a, b])

        for _, r in sub.iterrows():
            rk = float(r["risk_score"]) * 100
            win, wout, tx = str(r["wallet_in"]), str(r["wallet_out"]), str(r["txid"])
            add("win", win, rk); add("wout", wout, rk); add("tx", tx, rk)
            edge(win, tx); edge(tx, wout)
            if "src_ip" in r and pd.notna(r["src_ip"]):
                ip = str(r["src_ip"])
                add("ip", ip, rk)
                edge(ip, win)

        xs = {"ip": 90, "win": 320, "tx": 545, "wout": 770}
        nodes = []
        for col, members in cols.items():
            n = len(members)
            ys = np.linspace(60, 520, n) if n > 1 else [290.0]
            for y, (key, rk) in zip(ys, members.items()):
                kind = {"ip": "ip", "tx": "tx"}.get(col, "wallet")
                label = key if kind == "ip" else (f"tx {key[:6]}" if kind == "tx" else short(key))
                score = int(round(rk)) if kind != "wallet" else self._wallet_risk(key)
                nodes.append({"id": key, "label": label, "x": float(xs[col]), "y": float(y), "score": score,
                              "type": kind, "props": self._node_props(kind, key)})
        win = ["—", "—"]
        if len(sub):
            win = [sub["timestamp_chain"].min().strftime("%d %b %H:%M"), sub["timestamp_chain"].max().strftime("%d %b %H:%M")]
        return {"nodes": nodes, "edges": edges, "window": win, "stats": {"rows": int(len(sub)), "matched": int(len(sub))}}

    def trace(self, wallet: str, direction: str = "forward", hops: int = 4) -> dict:
        if wallet not in self.by_in and wallet not in self.by_out:
            if "…" in wallet:
                pre, suf = wallet.split("…")
                pool = set(self.by_in) | set(self.by_out)
                wallet = next((w for w in pool if str(w).startswith(pre) and str(w).endswith(suf)), wallet)
        df = self.df
        cur, visited = wallet, {wallet}
        path = [{"wallet": short(wallet), "amount": None, "risk": self._wallet_risk(wallet), "txid": None, "ts": None}]
        first_amt = None
        for _ in range(hops):
            pos = (self.by_in if direction == "forward" else self.by_out).get(cur, [])
            cand = df.iloc[pos]
            col = "wallet_out" if direction == "forward" else "wallet_in"
            cand = cand[~cand[col].isin(visited)]
            if cand.empty:
                break
            r = cand.sort_values("total_input_amount", ascending=False).iloc[0]
            cur = str(r[col])
            visited.add(cur)
            amt = float(r["total_input_amount"])
            first_amt = first_amt or amt
            path.append({"wallet": short(cur), "amount": round(amt, 4), "risk": self._wallet_risk(cur),
                         "txid": short(r["txid"]), "ts": r["timestamp_chain"].strftime("%Y-%m-%d %H:%M"),
                         "taint": int(round(100 * min(1.0, amt / first_amt)))})
        return {"path": [p["wallet"] for p in path], "hops": path}

    # ---------------------------------------------------------- correlation
    _OBS_TS = ["timestamp_network", "timestamp_net", "network_timestamp", "obs_timestamp",
               "timestamp_obs", "observed_at", "first_seen_network", "timestamp_observation"]

    @staticmethod
    def _pick(frame: pd.DataFrame, names: list[str]) -> str | None:
        return next((n for n in names if n in frame.columns), None)

    def _build_corr(self) -> None:
        """Normalise IP-observation <-> blockchain-tx matches into one frame:
        src_ip, asn, wallet_in, txid, ts_chain, ts_obs, latency, threat."""
        self.corr = None
        self.corr_source = "none"
        self.corr_derived = False
        self.obs_rows = None
        csv = os.path.join(DATA_DIR, "network_observations_1.csv")
        if os.path.exists(csv):
            try:
                with open(csv, "rb") as f:
                    self.obs_rows = max(sum(1 for _ in f) - 1, 0)
            except Exception:
                pass
        frames: list[tuple[pd.DataFrame, str]] = []
        cp = os.path.join(DATA_DIR, "correlated_data.pkl")
        if os.path.exists(cp):
            try:
                c = pd.read_pickle(cp)
                if isinstance(c, pd.DataFrame):
                    frames.append((c, "correlated_data.pkl"))
            except Exception:
                pass
        frames.append((self.df, "scored_data.pkl"))
        for fr, name in frames:
            ip = self._pick(fr, ["src_ip", "ip", "source_ip"])
            w = self._pick(fr, ["wallet_in", "wallet", "address"])
            tx = self._pick(fr, ["txid", "tx_id", "hash"])
            tc = self._pick(fr, ["timestamp_chain", "block_time"])
            to = self._pick(fr, self._OBS_TS)
            lat = self._pick(fr, ["latency_seconds", "latency"])
            if not (ip and w and tx and tc) or not (to or lat):
                continue
            
            # --- FIX: Explicitly convert and strip timezone metadata here ---
            ts_chain_series = pd.to_datetime(fr[tc])
            if ts_chain_series.dt.tz is not None:
                ts_chain_series = ts_chain_series.dt.tz_localize(None)

            out = pd.DataFrame({
                "src_ip": fr[ip].astype(str), 
                "wallet_in": fr[w].astype(str), 
                "txid": fr[tx].astype(str),
                "ts_chain": ts_chain_series,
            })
            # ----------------------------------------------------------------

            asn = self._pick(fr, ["asn", "src_asn"])
            out["asn"] = fr[asn].astype(str).values if asn else "—"
            thr = self._pick(fr, ["src_is_threat"])
            out["threat"] = fr[thr].fillna(0).astype(float).values if thr else 0.0
            
            if to:
                # --- FIX: Convert to datetime and strip timezone metadata here ---
                ts_obs_series = pd.to_datetime(fr[to])
                if ts_obs_series.dt.tz is not None:
                    ts_obs_series = ts_obs_series.dt.tz_localize(None)
                
                out["ts_obs"] = ts_obs_series.values
                out["latency"] = (out["ts_chain"] - out["ts_obs"]).dt.total_seconds()
            else:
                out["latency"] = fr[lat].astype(float).values
                out["ts_obs"] = out["ts_chain"] - pd.to_timedelta(out["latency"], unit="s")
                self.corr_derived = True
                
            self.corr = out[out["src_ip"].notna() & (out["src_ip"] != "nan")].reset_index(drop=True)
            self.corr_source = name
            return


    def correlation(self, limit: int = 20, bins: int = 48) -> dict | None:
        c = self.corr
        if c is None or c.empty:
            return None
        tau = max(float(c["latency"].abs().median()), 1.0)
        grp = (c.groupby(["src_ip", "wallet_in"])
                .agg(events=("txid", "size"), med_lat=("latency", "median"), threat=("threat", "mean"), asn=("asn", "first"))
                .reset_index())
        wa = c.groupby(["wallet_in", "asn"]).size().rename("wa").reset_index()
        wt = c.groupby("wallet_in").size().rename("wt").reset_index()
        ipt = c.groupby("src_ip").size().rename("ipt").reset_index()
        grp = (grp.merge(wa, on=["wallet_in", "asn"], how="left")
                  .merge(wt, on="wallet_in", how="left").merge(ipt, on="src_ip", how="left"))
        grp["time_proximity"] = 1.0 / (1.0 + grp["med_lat"].abs() / tau)
        grp["ip_cooccurrence"] = grp["events"] / grp["ipt"]
        grp["asn_consistency"] = grp["wa"] / grp["wt"]
        grp["threat_match"] = grp["threat"].clip(0, 1)
        grp["evidence_volume"] = 1.0 - np.exp(-grp["events"] / 3.0)
        grp["composite"] = (0.35 * grp["time_proximity"] + 0.25 * grp["ip_cooccurrence"] + 0.20 * grp["asn_consistency"]
                            + 0.10 * grp["threat_match"] + 0.10 * grp["evidence_volume"])
        top = grp.sort_values(["composite", "events"], ascending=False).head(limit)

        t0 = min(c["ts_obs"].min(), c["ts_chain"].min())
        t1 = max(c["ts_obs"].max(), c["ts_chain"].max())
        ns = lambda s_: pd.to_datetime(s_).values.astype("datetime64[ns]").astype("int64")  # noqa: E731
        edges_ = np.linspace(t0.value, t1.value, bins + 1)
        obs_h = np.histogram(ns(c["ts_obs"]), bins=edges_)[0]
        tx_h = np.histogram(ns(c["ts_chain"]), bins=edges_)[0]
        span = max(t1.value - t0.value, 1)
        pct = lambda t: float((pd.Timestamp(t).value - t0.value) / span * 100)  # noqa: E731

        cands, links = [], []
        keys = ["time_proximity", "ip_cooccurrence", "asn_consistency", "threat_match", "evidence_volume"]
        for i, r in enumerate(top.itertuples(index=False)):
            g = self.geo.get(r.src_ip, {})
            cands.append({
                "id": f"{r.src_ip}|{r.wallet_in}", "ip": r.src_ip, "asn": str(r.asn), "wallet": short(r.wallet_in),
                "wallet_full": r.wallet_in, "confidence": int(round(float(r.composite) * 100)), "events": int(r.events),
                "geo": g.get("code") or "—", "median_latency_s": round(float(r.med_lat), 2),
                "breakdown": {k: int(round(float(getattr(r, k)) * 100)) for k in keys},
            })
            if i < 5:
                ev = c[(c["src_ip"] == r.src_ip) & (c["wallet_in"] == r.wallet_in)].sort_values("latency", key=lambda x: x.abs()).iloc[0]
                links.append({"obs": pct(ev["ts_obs"]), "tx": pct(ev["ts_chain"]), "ip": r.src_ip, "wallet": short(r.wallet_in)})
        axis = [(t0 + (t1 - t0) * f).strftime("%d %b %H:%M") for f in (0, .25, .5, .75, 1)]
        return {
            "summary": {"events": int(len(c)), "pairs": int(len(grp)), "median_latency_s": round(tau, 2),
                        "raw_observations": self.obs_rows, "source": self.corr_source, "derived_obs_time": self.corr_derived},
            "lanes": {"obs": obs_h.tolist(), "tx": tx_h.tolist()},
            "axis": axis, "links": links, "candidates": cands,
        }