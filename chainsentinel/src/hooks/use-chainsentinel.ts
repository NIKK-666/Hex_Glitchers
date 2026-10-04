import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  api, type AlertsResp, type ClusterRow, type ExplainResp, type GeoRow, type GraphResp,
  type ModelCard, type Overview, type TraceResp, type CorrelationResp,
} from '@/lib/api';

// Every hook returns react-query state; App.tsx falls back to its built-in demo data
// when `data` is undefined (backend offline), so the UI never breaks.
const opts = { retry: 1, staleTime: 15_000, refetchOnWindowFocus: false } as const;

export const useOverview = () =>
  useQuery({ queryKey: ['overview'], queryFn: () => api<Overview>('/overview'), refetchInterval: 30_000, ...opts });

export const useAlerts = () =>
  useQuery({ queryKey: ['alerts'], queryFn: () => api<AlertsResp>('/alerts?limit=100'), ...opts });

export const useExplain = (txid?: string) =>
  useQuery({ queryKey: ['explain', txid], queryFn: () => api<ExplainResp>(`/explain/${txid}`), enabled: !!txid, ...opts });

export const useClusters = () =>
  useQuery({ queryKey: ['clusters'], queryFn: () => api<ClusterRow[]>('/clusters'), ...opts });

export const useGeo = () =>
  useQuery({ queryKey: ['geo'], queryFn: () => api<GeoRow[]>('/geo'), ...opts });

export const useModelCard = () =>
  useQuery({ queryKey: ['model'], queryFn: () => api<ModelCard>('/model'), ...opts });

export const useGraphData = (q: string, minRisk: number, rows: number) =>
  useQuery({
    queryKey: ['graph', q, minRisk, rows],
    queryFn: () => api<GraphResp>(`/graph?min_risk=${minRisk}&rows=${rows}${q ? `&q=${encodeURIComponent(q)}` : ''}`),
    placeholderData: keepPreviousData, ...opts,
  });

export const useCorrelation = () =>
  useQuery({ queryKey: ['correlation'], queryFn: () => api<CorrelationResp>('/correlation?limit=20'), ...opts });

export const useTrace = (wallet: string, direction: 'forward' | 'backward', hops: number) =>
  useQuery({
    queryKey: ['trace', wallet, direction, hops],
    queryFn: () => api<TraceResp>(`/trace?wallet=${encodeURIComponent(wallet)}&direction=${direction}&hops=${hops}`),
    enabled: wallet.length > 0, ...opts,
  });

export const useWhatIf = () =>
  useMutation({
    mutationFn: (b: { txid: string; overrides: Record<string, number> }) =>
      api<{ txid: string; risk_score: number; severity: string }>('/score', { method: 'POST', body: JSON.stringify(b) }),
  });

export const useSetStatus = () => {
  const qc = useQueryClient();
  return useMutation({
    onSuccess: () => qc.invalidateQueries({ queryKey: ['alerts'] }),
    mutationFn: (b: { id: string; status: string }) =>
      api(`/alerts/${b.id}/status`, { method: 'POST', body: JSON.stringify({ status: b.status }) }),
  });
};