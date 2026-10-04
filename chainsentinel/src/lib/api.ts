const API_BASE = 'http://localhost:8000/api';

export const api = {
  getOverview: () => fetch(`${API_BASE}/overview`).then(res => res.json()),
  getAlerts: (minRisk = 0.45) => fetch(`${API_BASE}/alerts?min_risk=${minRisk}`).then(res => res.json()),
  getAlertDetail: (txid: string) => fetch(`${API_BASE}/alerts/${txid}`).then(res => res.json()),
  getGraph: (q = '', minRisk = 0.45) => fetch(`${API_BASE}/graph?q=${q}&min_risk=${minRisk}`).then(res => res.json()),
  getClusters: () => fetch(`${API_BASE}/clusters`).then(res => res.json()),
  getCorrelation: () => fetch(`${API_BASE}/correlation`).then(res => res.json()),
  getTrace: (wallet: string, direction = 'forward', hops = 4) => 
    fetch(`${API_BASE}/trace?wallet=${wallet}&direction=${direction}&hops=${hops}`).then(res => res.json()),
};
