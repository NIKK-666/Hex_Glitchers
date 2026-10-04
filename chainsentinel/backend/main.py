"""ChainSentinel API.

Run from the folder that contains data2/:
    uvicorn main:app --reload --port 8000
(or set CS_DATA_DIR=/path/to/data2)
"""
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine import Engine

ENGINE: Engine | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global ENGINE
    ENGINE = Engine()  # loads models + runs the GraphSAGE forward pass once
    yield


app = FastAPI(title="ChainSentinel API", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def eng() -> Engine:
    if ENGINE is None:
        raise HTTPException(503, "Engine still loading")
    return ENGINE


@app.get("/api/health")
def health():
    e = eng()
    return {"ok": True, "rows": int(len(e.df)), "features": len(e.features), "geoip": e.geoip is not None}


@app.get("/api/overview")
def overview():
    return eng().overview()


@app.get("/api/alerts")
def alerts(min_risk: float = Query(0.45, ge=0, le=1), limit: int = Query(100, ge=1, le=500)):
    return eng().alerts(min_risk, limit)


@app.get("/api/alerts/{txid}")
def alert_detail(txid: str):
    a = eng().alert_by_txid(txid)
    if a is None:
        raise HTTPException(404, "Unknown txid")
    return a


class StatusBody(BaseModel):
    status: Literal["New", "Investigating", "Escalated", "False positive"]


@app.post("/api/alerts/{alert_id}/status")
def set_status(alert_id: str, body: StatusBody):
    eng().set_status(alert_id, body.status)
    return {"id": alert_id, "status": body.status}


@app.get("/api/explain/{txid}")
def explain(txid: str):
    r = eng().explain(txid)
    if r is None:
        raise HTTPException(404, "Unknown txid")
    return r


class WhatIf(BaseModel):
    txid: str
    overrides: dict[str, float]


@app.post("/api/score")
def score(body: WhatIf):
    r = eng().what_if(body.txid, body.overrides)
    if r is None:
        raise HTTPException(404, "Unknown txid")
    return r


@app.get("/api/clusters")
def clusters():
    return eng().clusters()


@app.get("/api/geo")
def geo():
    return eng().geo_summary()


@app.get("/api/model")
def model():
    return eng().model_card()


@app.get("/api/graph")
def graph(q: str | None = None, min_risk: float = Query(0.45, ge=0, le=1), rows: int = Query(15, ge=3, le=40)):
    return eng().subgraph(q, min_risk, rows)


@app.get("/api/correlation")
def correlation(limit: int = Query(20, ge=1, le=100)):
    r = eng().correlation(limit)
    if r is None:
        raise HTTPException(404, "No usable correlation data (need src_ip, wallet_in, txid, timestamp_chain + latency or network timestamp)")
    return r


@app.get("/api/trace")
def trace(wallet: str, direction: Literal["forward", "backward"] = "forward", hops: int = Query(4, ge=1, le=8)):
    return eng().trace(wallet, direction, hops)