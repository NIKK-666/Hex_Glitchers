# ChainSentinel

**Offline AI-powered monitoring and investigative-lead generation for Bitcoin transaction fraud**

Smart India Hackathon 2026 — SIH26146 — Theme: Blockchain & Cybersecurity

---

## Table of contents

- [Problem](#problem)
- [Solution](#solution)
- [Idea / Approach](#idea--approach)
- [Architecture](#architecture)
- [Feasibility](#feasibility)
- [Viability](#viability)
- [Impact and benefits](#impact-and-benefits)
- [Comparison: existing vs proposed solutions](#comparison-existing-vs-proposed-solutions)
- [Repository structure](#repository-structure)
- [References](#references)

---

## Problem

Bitcoin's pseudonymous, peer-to-peer design lets criminal actors move, layer, and cash out illicit funds — ransomware payments, darknet-market proceeds, extortion, and laundering — while evading traditional financial surveillance. A wallet address alone reveals nothing about its owner, but a transaction is never purely on-chain: broadcasting it requires a real P2P connection over a real IP address. That single crack between the network layer and the blockchain layer is largely unexploited by existing tooling, and investigators are left correlating bulk transaction dumps and network logs by hand, with no automated way to link the two, detect anomalous behavior, or explain why a given wallet is suspicious.

## Solution

**ChainSentinel** ingests bulk Bitcoin transaction and P2P network metadata (CSV/JSON/XML), correlates network-layer observations (source/destination IP, port, timing) with blockchain-layer data (wallet addresses, TXIDs, amounts), and applies a hybrid AI/ML pipeline to detect anomalies, cluster entities, and generate a ranked, explainable list of investigative leads — entirely offline, on a single Linux machine, with zero internet dependency at runtime.

It does not replace analyst judgment; it compresses the search space. Instead of manually tracing a transaction graph with thousands of nodes, an analyst opens a ranked alert queue where every flagged wallet already carries a confidence score, a fraud typology, and a plain-language explanation of exactly which features drove the score.

## Idea / Approach

### Challenge objectives addressed

- Ingest and parse bulk metadata: timestamp, src/dst IP and port, TXID, input/output addresses, amounts, fee, script type
- Build an entity/transaction graph linking IPs, wallets, and transactions
- Implement a working AI/ML model, not a rules engine
- Generate a ranked, explainable alert list with a confidence score per flag
- Present findings through an interactive dashboard / link-analysis visualization

### Detection pipeline

1. **Correlation** — link each transaction's P2P broadcast origin (IP/port/timing) to its on-chain footprint (TXID/wallet/amount)
2. **Entity resolution** — common-input-ownership clustering groups wallets likely controlled by the same real-world actor
3. **Structural pattern matching** — rule-based detection of known laundering shapes: peeling chains, CoinJoin/mixer structures, circular ring layering
4. **GraphSAGE** — inductive node embeddings learned from each wallet's local transaction neighborhood; because it is inductive (not transductive), it scores a wallet it has never seen before without retraining — essential for a system that cannot call out for live updates
5. **XGBoost** — fuses GraphSAGE embeddings with handcrafted transaction features (fee, fan-in/fan-out, amount statistics, GeoIP risk) into a calibrated fraud probability and typology tag (ring / peel chain / mixer hub / ransomware collector / darknet payment)
6. **Isolation Forest + HDBSCAN** — an unsupervised layer running on the same GraphSAGE embeddings, catching structurally anomalous wallets that do not match *any* labeled typology, then grouping similar anomalies into candidate **novel fraud patterns** for analyst review, rather than discarding them
7. **SHAP (TreeExplainer) + plain-language summary** — every flagged entity carries an exact, per-feature contribution breakdown and a human-readable explanation of why it was flagged, satisfying the problem statement's explicit explainability requirement

### Why this combination, specifically

A purely supervised model only recognizes fraud typologies present in its training labels. Real laundering techniques evolve faster than any labeled dataset can keep up with, so XGBoost's known-typology detection is paired with an unsupervised anomaly layer running on the exact same embeddings, at near-zero additional pipeline cost. The result is a system that can say *"this doesn't match anything we've labeled, but it is structurally unusual"* instead of silently missing it.

## Architecture

```
NTRO LOCAL SYSTEM (air-gapped, no internet egress required)

 Bulk Bitcoin transaction + P2P metadata (CSV / JSON / XML)
                      │
                      ▼
         Ingestion & validation (Python)
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
   Data ingestion            Correlation engine
   (parser, offline            (TXID + timestamp
   GeoIP lookup)                matching: network
         │                       layer ↔ blockchain layer)
         └────────────┬────────────┘
                      ▼
       ┌──────────────┴──────────────┐
       ▼                             ▼
  PostgreSQL                    Neo4j (graph database)
  (wallets, typology tags,      (Address / Transaction / IP
   case notes; JSONB:            nodes; SENDS / PAYS /
   SHAP + LLM payloads)           ORIGINATED_FROM edges)
                                      │
                                      ▼
                           Entity clustering
                      (common-input-ownership heuristic)
                         │                    │
                         ▼                    ▼
              Structural pattern        GraphSAGE
              matching (peel chains,    (inductive node
              mixers, CoinJoin)          embeddings)
                         │                    │
                         │        ┌───────────┴───────────┐
                         │        ▼                       ▼
                         │    XGBoost                Isolation Forest
                         │  (known typology,          + HDBSCAN
                         │   confidence score)        (novel-pattern
                         │        │                    discovery)
                         │        └───────────┬───────────┘
                         │                    ▼
                         │       Risk score + fraud typology
                         │   (SHAP explanation + LLM summary
                         │    → PostgreSQL; score → Neo4j)
                         └────────────┬───────────┘
                                      ▼
                     Security & deployment
              (fully offline, Linux-hosted, no external calls)
                                      ▼
                    Analyst dashboard (interactive)
         entity-link graph · ranked alerts · typology search
         · SHAP + LLM explain panel · peeling-chain / mixer view
```
<img width="962" height="973" alt="26146 pptx" src="https://github.com/user-attachments/assets/eb3fc76f-54c2-4e2f-a1ab-ddb184c187fb" />


**Model training** happens on a separate, non-air-gapped machine using a synthetic dataset modeled on real Bitcoin P2P/transaction fields (no seized or live-intercept data required for the MVP). Only signed, checksummed model artifacts (GraphSAGE weights, XGBoost model file) cross into the production air-gapped environment via approved offline media — raw training data never does.

## Feasibility

**Technical**
- Runs fully offline on a single mid-range Linux server (16–32 GB RAM); CPU-only inference is viable, GPU is optional and only used for periodic retraining
- Air-gapped packaging: Python dependencies as offline wheel bundles, GeoIP as a bundled `.mmdb` file, Docker images transferred via `docker save` / `docker load`
- Built entirely on mature, well-documented open-source components: PyTorch Geometric (GraphSAGE), XGBoost, scikit-learn (Isolation Forest), HDBSCAN, SHAP, Neo4j, PostgreSQL

**Economic**
- Fully open-source stack — no licensing cost
- No cloud or GPU spend required for inference
- Low maintenance overhead — no recurring API or data-subscription fees

## Viability

- **Offline model updates** — retraining happens on a separate, connected rig using updated (eventually real, labeled) data; resulting artifacts are reviewed, signed, and moved into production via approved offline media, preserving the air-gap
- **GeoIP refresh** — updated `.mmdb` files are downloaded once on a connected machine, validated, then pushed in through the same controlled transfer process
- **Scalable rollout** — single-node pilot → multi-node LAN cluster (sharded graph processing) → integration with existing case-management systems
- **Long-term value** — GraphSAGE's inductive design means the trained model keeps scoring new wallets correctly as the transaction graph grows, reducing how often a full retrain is needed; the unsupervised layer means new laundering techniques can be surfaced for review even before they are formally labeled

## Impact and benefits

- **Faster triage** — automated, graph-aware scoring replaces manual chain-tracing for the majority of cases
- **Broader coverage** — catches both known typologies (via XGBoost) and structurally novel patterns nobody has labeled yet (via Isolation Forest + HDBSCAN)
- **Evidence-grade explainability** — every alert carries an exact SHAP feature breakdown and a plain-language summary, giving investigators a traceable, defensible basis for action
- **Deployable where it matters most** — fully offline operation means it can run inside an air-gapped, security-cleared environment with no architectural compromise
- **Low-cost to pilot** — built and demoed on synthetic, open-source-modeled data and entirely open-source tooling

## Comparison: existing vs proposed solutions

| | Existing approaches | ChainSentinel (proposed) |
|---|---|---|
| Detection basis | Static blacklist / whitelist matching | AI/ML-driven detection of anomalous and suspicious transaction patterns |
| Monitoring | Limited real-time monitoring | Continuous monitoring of Bitcoin transaction traffic |
| Analyst workload | Requires analysts to inspect large volumes of transaction data manually | Automated, AI-powered transaction analysis with prioritized output |
| Adaptability | Limited ability to identify new or evolving attack patterns | Learns transaction behavior and surfaces emerging/novel threats via unsupervised anomaly detection |
| Methodology | Relies mainly on manual analysis and predefined rules | Combines structural rules, supervised ML (GraphSAGE + XGBoost), and unsupervised anomaly detection |
| Prioritization | No automated prioritization | Automatically processes and prioritizes high-risk transactions by confidence score |
| Explainability | Rule match or analyst judgment only | SHAP-based, per-feature explanation plus plain-language summary for every flag |

## Repository structure

```
data/           synthetic dataset generator, GeoIP lookup helper
db/             Neo4j and PostgreSQL access layers
pipeline/       ingestion pipeline (writes to both stores)
graph/          Neo4j → PyTorch Geometric graph construction
models/         GraphSAGE encoder, XGBoost fusion, unsupervised anomaly detection
explain/        SHAP + plain-language explanation, fused typology routing
dashboard/      analyst-facing interface
docker-compose.yml   offline Neo4j + PostgreSQL services
requirements.txt
```

## References

- Hamilton, Ying, Leskovec. [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) (GraphSAGE)
- Demystifying Fraudulent Transactions and Illicit Nodes in the Bitcoin Network (KDD 2023)
- Anti-Money Laundering in Bitcoin
- [GeoLite2 free geolocation databases](https://dev.maxmind.com/geoip/geolite2-free-geolocation-data)
- [Fraud Detection on Bitcoin Transaction Graphs Using Graph Convolutional Networks](https://medium.com/stanford-cs224w/fraud-detection-on-bitcoin-transaction-graphs-using-graph-convolutional-networks-5fc50a903687)
- [Transaction Fraud Detection via Spatial-Temporal-Aware Graph Transformer](https://arxiv.org/pdf/2307.05121)
- [Towards Quantum-Ready Blockchain Fraud Detection via Ensemble Graph Neural Networks](https://arxiv.org/pdf/2509.23101)
- [Unsupervised clustering of Bitcoin transactions](https://link.springer.com/article/10.1186/s40854-023-00525-y)
- [Wallet and Entity Identification in Blockchain Analytics](https://blog.amlbot.com/wallet-and-entity-identification-in-blockchain-analytics/)
- [Bitcoin address clustering method based on multiple heuristic conditions](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/blc2.12014)
