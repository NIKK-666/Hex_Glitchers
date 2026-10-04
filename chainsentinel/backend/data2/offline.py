"""
SIH 2026 - Problem Statement 26146
AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic
Complete Data Ingestion, Neo4j Sync, & AI Inference Backend Pipeline
"""

import os
import json
import xml.etree.ElementTree as ET
import csv
import pickle
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from neo4j import GraphDatabase

# Machine Learning & Deep Learning Frameworks
import torch
import torch.nn as nn
import xgboost as xgb
import shap

# ---------------------------------------------------------
# GLOBAL CONSTANTS & FILE ARTIFACT PATHS (From RAR Workspace)
# ---------------------------------------------------------
BLOCKCHAIN_JSON = "blockchain_transactions_1.json"
EXTERNAL_XML = "external_asn_threats_1.xml"
NETWORK_CSV = "network_observations_1.csv"

# Model and Archive Artifact Files
GRAPHSAGE_MODEL_PATH = "graphsage_encoder.pt"
XGBOOST_MODEL_PATH = "xgboost_forensics_model.pkl"
SHAP_EXPLAINER_PATH = "shap_explainer.pkl"
SCORED_DATA_PKL = "scored_data.pkl"

# Connection strings for offline databases on localhost
PG_CONN_STRING = "host=localhost dbname=sih_bitcoin_db user=postgres password=yourpassword port=5432"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "yourpassword"

# ---------------------------------------------------------
# NEURAL NETWORK ARCHITECTURE DEFINITION
# ---------------------------------------------------------
class GraphSAGEEncoder(nn.Module):
    """
    Local implementation matching graphsage_encoder.pt structure
    """
    def __init__(self, in_feats=5, out_feats=4):
        super(GraphSAGEEncoder, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.linear(x))

# ---------------------------------------------------------
# PHASE 1: PARSE AND RAW DATASET INGESTION ENGINE
# ---------------------------------------------------------
def initialize_database_schemas():
    """Initializes tables and indexing arrays for raw ingestion passes."""
    print("[*] Initializing PostgreSQL schemas...")
    pg_conn = psycopg2.connect(PG_CONN_STRING)
    pg_cur = pg_conn.cursor()
    
    pg_cur.execute("""
        CREATE TABLE IF NOT EXISTS threat_feed (
            ip VARCHAR(50) PRIMARY KEY, asn VARCHAR(20), 
            risk_category VARCHAR(100), threat_level VARCHAR(20)
        );
        CREATE TABLE IF NOT EXISTS network_traffic (
            id SERIAL PRIMARY KEY, timestamp TIMESTAMP, src_ip VARCHAR(50), 
            dst_ip VARCHAR(50), src_port INT, dst_port INT, txid VARCHAR(100)
        );
        CREATE TABLE IF NOT EXISTS blockchain_transactions (
            txid VARCHAR(100) PRIMARY KEY, timestamp TIMESTAMP, fee NUMERIC(20, 8),
            script_type VARCHAR(20), label INT, pattern_type VARCHAR(50), 
            cluster_id VARCHAR(100), risk_score REAL, is_illicit BOOLEAN
        );
        CREATE INDEX IF NOT EXISTS idx_traffic_txid ON network_traffic (txid);
    """)
    pg_conn.commit()
    pg_cur.close()
    pg_conn.close()

def parse_and_load_all_sources():
    """Ingests JSON, XML, and CSV files directly into the databases."""
    print("[*] Starting extraction and multi-source data ingestion pipeline...")
    
    # A. Ingest Threat XML
    if os.path.exists(EXTERNAL_XML):
        tree = ET.parse(EXTERNAL_XML)
        root = tree.getroot()
        threat_records = []
        for host in root.findall('Host'):
            threat_records.append((
                host.find('IP').text, host.find('ASN').text, 
                host.find('RiskCategory').text, host.find('ThreatLevel').text
            ))
        pg_conn = psycopg2.connect(PG_CONN_STRING)
        with pg_conn.cursor() as cur:
            execute_values(cur, "INSERT INTO threat_feed VALUES %s ON CONFLICT (ip) DO NOTHING;", threat_records)
        pg_conn.commit()
        pg_conn.close()
        print(f"[+] Loaded {len(threat_records)} XML threat definitions.")

    # B. Ingest CSV Logs
    if os.path.exists(NETWORK_CSV):
        traffic_records = []
        with open(NETWORK_CSV, mode='r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header row
            for row in reader:
                if len(row) >= 6:
                    traffic_records.append((row[0], row[1], row[2], int(row[3]), int(row[4]), row[5]))
        pg_conn = psycopg2.connect(PG_CONN_STRING)
        with pg_conn.cursor() as cur:
            execute_values(cur, "INSERT INTO network_traffic (timestamp, src_ip, dst_ip, src_port, dst_port, txid) VALUES %s;", traffic_records)
        pg_conn.commit()
        pg_conn.close()
        print(f"[+] Ingested {len(traffic_records)} raw CSV transaction observations.")

    # C. Ingest Blockchain JSON + Populate Graph Structural Intersections
    if os.path.exists(BLOCKCHAIN_JSON):
        with open(BLOCKCHAIN_JSON, 'r') as f:
            tx_data = json.load(f)
            
        pg_batch = []
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        with neo4j_driver.session() as session:
            for tx in tx_data:
                txid = tx["txid"]
                pg_batch.append((txid, tx["timestamp"], tx["fee"], tx["script_type"], tx["label"], tx["pattern_type"], tx.get("cluster_id")))
                
                # Execute Unrolled Cypher to link addresses through transaction hubs
                session.run("""
                    MERGE (t:Transaction {txid: $txid})
                    SET t.timestamp = $timestamp, t.fee = $fee, t.pattern_type = $pattern_type
                    WITH t
                    UNWIND $inputs AS input_addr
                    MERGE (in_a:Address {address: input_addr})
                    CREATE (in_a)-[:SENDS_TO]->(t)
                    WITH t
                    UNWIND $outputs AS output_addr
                    MERGE (out_a:Address {address: output_addr})
                    CREATE (t)-[:SENDS_TO]->(out_a)
                """, txid=txid, timestamp=tx["timestamp"], fee=tx["fee"], pattern_type=tx["pattern_type"],
                     inputs=tx["input_addresses"], outputs=tx["output_addresses"])
                     
        pg_conn = psycopg2.connect(PG_CONN_STRING)
        with pg_conn.cursor() as cur:
            execute_values(cur, "INSERT INTO blockchain_transactions (txid, timestamp, fee, script_type, label, pattern_type, cluster_id) VALUES %s ON CONFLICT (txid) DO NOTHING;", pg_batch)
        pg_conn.commit()
        pg_conn.close()
        neo4j_driver.close()
        print(f"[+] Complete structural indexing mapped for {len(pg_batch)} blockchain logs.")

# ---------------------------------------------------------
# PHASE 2: EXTRACT GDS GRAPH METRICS OUT OF NEO4J
# ---------------------------------------------------------
def run_and_pull_graph_analytics():
    print("[*] Initializing Neo4j Graph Data Science execution blocks...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    with driver.session() as session:
        # Re-initialize memory projected graph mapping space safely
        session.run("CALL gds.graph.drop('bitcoin_network', false) YIELD graphName")
        session.run("""
            CALL gds.graph.project(
              'bitcoin_network',
              ['Address', 'Transaction'],
              { SENDS_TO: { orientation: 'NATURAL' } }
            )
        """)
        
        # Calculate PageRank
        print("[*] Stream extraction for calculated PageRank features...")
        pr_result = session.run("""
            CALL gds.pageRank.stream('bitcoin_network') YIELD nodeId, score
            RETURN gds.util.asNode(nodeId).address AS address, score AS pagerank
        """)
        pr_df = pd.DataFrame([dict(r) for r in pr_result]).dropna()
        
        # Calculate Community Detection (Louvain)
        print("[*] Stream extraction for Louvain community IDs...")
        lv_result = session.run("""
            CALL gds.louvain.stream('bitcoin_network') YIELD nodeId, communityId
            RETURN gds.util.asNode(nodeId).address AS address, communityId
        """)
        lv_df = pd.DataFrame([dict(r) for r in lv_result]).dropna()
        
        # Clean cached projection
        session.run("CALL gds.graph.drop('bitcoin_network')")
        
    driver.close()
    merged_graph_features = pd.merge(pr_df, lv_df, on="address", how="inner")
    return merged_graph_features

# ---------------------------------------------------------
# PHASE 3: CORRELATION SYNCHRONIZATION AND DATA PACKING
# ---------------------------------------------------------
def merge_and_build_features_dataframe(graph_features_df):
    print("[*] Merging structural vector spaces into PostgreSQL storage indexes...")
    pg_conn = psycopg2.connect(PG_CONN_STRING)
    pg_cur = pg_conn.cursor()
    
    pg_cur.execute("""
        CREATE TABLE IF NOT EXISTS address_graph_metrics (
            address VARCHAR(100) PRIMARY KEY, pagerank REAL, community_id BIGINT, last_updated TIMESTAMP DEFAULT NOW()
        );
    """)
    pg_conn.commit()
    
    records = list(graph_features_df.itertuples(index=False, name=None))
    upsert_query = """
        INSERT INTO address_graph_metrics (address, pagerank, community_id) VALUES %s
        ON CONFLICT (address) DO UPDATE SET pagerank = EXCLUDED.pagerank, community_id = EXCLUDED.community_id;
    """
    execute_values(pg_cur, upsert_query, records)
    pg_conn.commit()
    
    # Generate full dataset matrix using PostgreSQL for analytical calculations
    query = """
        SELECT b.txid, b.fee, b.label, m.pagerank, m.community_id
        FROM blockchain_transactions b
        LEFT JOIN network_traffic n ON b.txid = n.txid
        LEFT JOIN address_graph_metrics m ON n.src_ip = m.address OR n.dst_ip = m.address;
    """
    dataset_df = pd.read_sql_query(query, pg_conn)
    pg_cur.close()
    pg_conn.close()
return dataset_df
# ---------------------------------------------------------
# PHASE 4: OFFLINE MULTI-MODEL AI EVALUATION
# ---------------------------------------------------------
def run_ai_forensics_inference(dataset_df):
    if dataset_df.empty:
        print("[-] Target data matrix empty. Stopping inference pass.")
        return None
    print("[*] Running multi-layer AI models safely offline...")
    # 1. Evaluate Structure Using PyTorch GraphSAGE Encoder
    encoder = GraphSAGEEncoder(in_feats=5, out_feats=4)
    if os.path.exists(GRAPHSAGE_MODEL_PATH):
        encoder.load_state_dict(torch.load(GRAPHSAGE_MODEL_PATH, weights_only=True))
        encoder.eval()
        X_tab = dataset_df[['fee', 'pagerank', 'community_id']].fillna(0).values
        dummy_input_features = torch.tensor(np.hstack([X_tab, np.zeros((X_tab.shape[0], 2))]), dtype=torch.float32)
        with torch.no_grad():
        graph_embeddings = encoder(dummy_input_features).numpy()
        # Combine tabular dimensions with structural node embeddings
        X_complete = np.hstack([X_tab, graph_embeddings])
        # 2. Run Classification with local XGBoost model
        with open(XGBOOST_MODEL_PATH, 'rb') as f:
        xgb_classifier = pickle.load(f)
        probabilities = xgb_classifier.predict_proba(X_complete)[:, 1]
        dataset_df['fraud_risk_score'] = probabilities
        # 3. Export calculations to a local data frame snapshot pickle
        print(f"[*] Caching processed calculations directly to storage artifacts: {SCORED_DATA_PKL}")
        with open(SCORED_DATA_PKL, 'wb') as f:
        pickle.dump(dataset_df, f)
        return dataset_df
# ---------------------------------------------------------
# PHASE 5: DUAL-DATABASE FLAG SYNCHRONIZATION
# ---------------------------------------------------------
def write_back_risk_scores_to_databases(scored_df):
    print("[*] Committing finalized classification metrics back across databases...")
    # A. Commit to PostgreSQL
    pg_conn = psycopg2.connect(PG_CONN_STRING)
    pg_cur = pg_conn.cursor()
    update_records = [(float(r.fraud_risk_score), bool(r.fraud_risk_score > 0.75), r.txid) for r in scored_df.itertuples()]
    update_query = "UPDATE blockchain_transactions SET risk_score = %s, is_illicit = %s WHERE txid = %s;"
    execute_values(pg_cur, update_query, update_records, template="(%s, %s, %s)")
    pg_conn.commit()
    pg_cur.close()
    pg_conn.close()
    # B. Commit to Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
    tx_list = [{"txid": r.txid, "score": float(r.fraud_risk_score)} for r in scored_df.itertuples()]
    session.run("""
    UNWIND $batch AS data
    MATCH (t:Transaction {txid: data.txid})
    SET t.risk_score = data.score,
    t.status = CASE WHEN data.score > 0.75 THEN 'ILLICIT' ELSE 'LICIT' END
    """, batch=tx_list)
    driver.close()
    print("[+] Entire data processing pipeline and model evaluation sync completed successfully.")
# ---------------------------------------------------------
# SYSTEM MAIN ENTRY ROUTINE
# ---------------------------------------------------------
if name == "main":
    initialize_database_schemas()
    parse_and_load_all_sources()
    # Run algorithms and compile unified analytical data matrix
    graph_features = run_and_pull_graph_analytics()
    unified_dataframe = merge_and_build_features_dataframe(graph_features)
    # Run PyTorch and XGBoost inferences
    final_scored_df = run_ai_forensics_inference(unified_dataframe)
# Synchronize tracking updates
if final_scored_df is not None:
    write_back_risk_scores_to_databases(final_scored_df)
