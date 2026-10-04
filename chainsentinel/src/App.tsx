import { Fragment, type ReactNode, createContext, useContext, useEffect, useMemo, useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ErrorBoundary } from '@/components/error-boundary';
import { Toaster, toast } from 'sonner';
import { TooltipProvider } from '@/components/ui/tooltip';
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, Pie, PieChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import {
  Activity, AlertTriangle, ArrowDownLeft, ArrowLeftRight, ArrowUpRight, BarChart3, Bell, Boxes,
  Check, ChevronDown, ChevronRight, CircleAlert, CircleCheck, CircleDot, Clock3, Command,
  Cpu, Crosshair, Database, Download, Eye, FileSearch, FileText, GitBranch, GitMerge, Globe2,
  Layers3, ListFilter, Menu, Moon, Network, Pause, Play, Radio, RefreshCw, Search, Settings,
  ShieldAlert, SlidersHorizontal, Sparkles, Sun, Timer, Upload, WalletCards, Workflow, X, Zap,
} from 'lucide-react';
import { Link, Route, Switch, Router as WouterRouter, useLocation } from 'wouter';
import NotFound from '@/pages/not-found';


// =====================================================================
// LIVE BACKEND ENDPOINT SPECIFICATIONS
// =====================================================================
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

type ThemeMode = 'dark' | 'light';

type ThemeContextType = {
  theme: ThemeMode;
  toggleTheme: () => void;
};

const ThemeContext = createContext<ThemeContextType>({
  theme: 'dark',
  toggleTheme: () => {},
});



const queryClient = new QueryClient();
const API_BASE = 'http://localhost:8000/api';
const telemetryData = [
  { time: '00:00', tx: 180, network: 120, flagged: 12 },
  { time: '02:00', tx: 210, network: 145, flagged: 15 },
  { time: '04:00', tx: 195, network: 138, flagged: 13 },
  { time: '06:00', tx: 260, network: 180, flagged: 18 },
  { time: '08:00', tx: 320, network: 220, flagged: 24 },
  { time: '10:00', tx: 410, network: 280, flagged: 31 },
  { time: '12:00', tx: 520, network: 360, flagged: 47 },
  { time: '14:00', tx: 460, network: 315, flagged: 38 },
  { time: '16:00', tx: 390, network: 270, flagged: 29 },
  { time: '18:00', tx: 430, network: 300, flagged: 34 },
  { time: '20:00', tx: 350, network: 240, flagged: 25 },
  { time: '22:00', tx: 290, network: 200, flagged: 19 },
  { time: '24:00', tx: 240, network: 165, flagged: 16 },
];
const riskDistribution = [
  { name: 'Critical', value: 8, color: '#ef4f79' },
  { name: 'High', value: 18, color: '#f472d0' },
  { name: 'Medium', value: 42, color: '#ffe23a' },
  { name: 'Low', value: 68, color: '#2fb8ff' },
  { name: 'Minimal', value: 40, color: '#34D399' },
];

const typologyData = [
  { name: 'Mixer / tumbler', value: 31, color: '#ef4f79' },
  { name: 'Layering / structuring', value: 24, color: '#f472d0' },
  { name: 'Exchange cash-out', value: 16, color: '#ffe23a' },
  { name: 'Ransomware collection', value: 12, color: '#2fb8ff' },
  { name: 'Darknet market', value: 18, color: '#34D399' },
  { name: 'Extortion / sextortion', value: 9, color: '#8b5cf6' },
];

const alerts = [
  {
    id: 'ALT-001',
    title: 'High-risk transaction cluster detected',
    severity: 'critical',
    status: 'open',
    timestamp: new Date().toISOString(),
    contributing_signals: [
      'Rapid fund movement',
      'Mixer interaction',
      'Multiple intermediary wallets',
    ],
  },
  {
    id: 'ALT-002',
    title: 'Suspicious exchange cash-out pattern',
    severity: 'high',
    status: 'open',
    timestamp: new Date().toISOString(),
    contributing_signals: [
      'Large outbound transfer',
      'Exchange interaction',
      'Velocity anomaly',
    ],
  },
  {
    id: 'ALT-003',
    title: 'Layering pattern identified',
    severity: 'medium',
    status: 'open',
    timestamp: new Date().toISOString(),
    contributing_signals: [
      'Multiple hops',
      'Short holding periods',
    ],
  },
  {
    id: 'ALT-004',
    title: 'Darknet-linked address activity',
    severity: 'high',
    status: 'open',
    timestamp: new Date().toISOString(),
    contributing_signals: [
      'Known darknet exposure',
      'Obfuscated transaction path',
    ],
  },
];

const geoData = [
  { country: 'United States', value: 32 },
  { country: 'United Kingdom', value: 18 },
  { country: 'Germany', value: 14 },
  { country: 'India', value: 12 },
  { country: 'Singapore', value: 9 },
  { country: 'Other', value: 15 },
];

const transactions = [
  {
    id: 'TX-001',
    hash: '0x8f2a...91cd',
    from: '0x7a21...3f91',
    to: '0x9bc2...7a10',
    amount: 125000,
    asset: 'USDT',
    timestamp: new Date().toISOString(),
    risk: 'high',
  },
  {
    id: 'TX-002',
    hash: '0x21bc...8e42',
    from: '0x4d91...22ab',
    to: '0x8ac1...9f31',
    amount: 84000,
    asset: 'USDC',
    timestamp: new Date().toISOString(),
    risk: 'medium',
  },
  {
    id: 'TX-003',
    hash: '0x73de...11fa',
    from: '0x91ab...71cd',
    to: '0x31ef...aa92',
    amount: 216000,
    asset: 'ETH',
    timestamp: new Date().toISOString(),
    risk: 'critical',
  },
];

const entities = [
  {
    id: 'ENT-001',
    name: '0x7a21...3f91',
    type: 'Wallet',
    risk: 'critical',
    score: 94,
  },
  {
    id: 'ENT-002',
    name: '0x9bc2...7a10',
    type: 'Wallet',
    risk: 'high',
    score: 82,
  },
  {
    id: 'ENT-003',
    name: 'Exchange Alpha',
    type: 'Exchange',
    risk: 'medium',
    score: 61,
  },
  {
    id: 'ENT-004',
    name: '0x4d91...22ab',
    type: 'Wallet',
    risk: 'high',
    score: 77,
  },
  {
    id: 'ENT-005',
    name: 'Mixer Cluster',
    type: 'Cluster',
    risk: 'critical',
    score: 97,
  },
];

const api = {
  getOverview: () => fetch(`${API_BASE}/overview`).then(res => {
    if (!res.ok) throw new Error('Engine unreachable');
    return res.json();
  }),
  getAlerts: (minRisk = 0.45) => fetch(`${API_BASE}/alerts?min_risk=${minRisk}`).then(res => res.json()),
  getAlertDetail: (txid: string) => fetch(`${API_BASE}/alerts/${txid}`).then(res => res.json()),
  updateAlertStatus: (alertId: string, status: string) => 
    fetch(`${API_BASE}/alerts/${alertId}/status`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status })
    }).then(res => res.json()),
  getGraph: (
  q = '',
  minRisk = 0.45,
  maxRows = 15
) => {
  const params = new URLSearchParams({
    q,
    min_risk: String(minRisk),
    max_rows: String(maxRows),
  });

  return fetch(`${API_BASE}/graph?${params.toString()}`)
    .then(res => {
      if (!res.ok) {
        throw new Error(`Graph API error: ${res.status}`);
      }
      return res.json();
    });
},
  getClusters: () => fetch(`${API_BASE}/clusters`).then(res => res.json()),
  getGeo: () => fetch(`${API_BASE}/geo`).then(res => res.json()),
  getCorrelation: () => fetch(`${API_BASE}/correlation`).then(res => res.json()),
  getTrace: (wallet: string, direction = 'forward', hops = 4) => 
    fetch(`${API_BASE}/trace?wallet=${wallet}&direction=${direction}&hops=${hops}`).then(res => res.json()),
};

const navItems = [
  { href: '/overview', label: 'Overview', icon: Activity }, 
  { href: '/alerts', label: 'Alerts', icon: ShieldAlert, count: 4 },
  { href: '/graph', label: 'Graph explorer', icon: Network }, 
  { href: '/tracer', label: 'Hop tracer', icon: ArrowLeftRight },
  { href: '/clusters', label: 'Entities & clusters', icon: Boxes }, 
  { href: '/correlation', label: 'Network correlation', icon: Radio }, 
  { href: '/geo', label: 'Geo map', icon: Globe2 },
];
const opsItems = [
  { href: '/cases', label: 'Cases', icon: FileSearch }, 
  { href: '/pipeline', label: 'Pipeline & model', icon: Workflow },
];



function riskLevel(score: number): Risk {
  return score >= 90 ? 'critical' : score >= 70 ? 'high' : score >= 40 ? 'medium' : 'low';
}
function RiskBadge({ score, compact = false }: { score: number; compact?: boolean }) {
  const level = riskLevel(score);
  return <span className={`risk risk-${level}`} data-testid={`risk-${score}`}><i className="risk-dot" />{!compact && <span>{level}</span>}<span>{score}</span></span>;
}
function StatusPill({ status }: { status: string }) {
  const cls = status.toLowerCase().replace(/\s+/g, '-');
  return <span className={`status-pill status-${cls}`}>{status}</span>;
}
function Panel({ title, note, action, children, className = '' }: { title: string; note?: string; action?: ReactNode; children: ReactNode; className?: string }) {
  return <section className={`card soc-panel ${className}`}><div className="card-head"><div><h2 className="section-title">{title}</h2>{note && <div className="section-note mt-1">{note}</div>}</div>{action}</div>{children}</section>;
}
function MiniSpark({ points, color = '#2fb8ff' }: { points: number[]; color?: string }) {
  const activePoints = points && points.length > 0 ? points : [0];

  const max = Math.max(...activePoints);
  const min = Math.min(...activePoints);
  const path = activePoints.map((value, index) => `${index ? 'L' : 'M'} ${(index / (activePoints.length - 1 || 1)) * 100} ${32 - ((value - min) / (max - min || 1)) * 26}`).join(' ');

  return <svg className="mini-spark" viewBox="0 0 100 36" preserveAspectRatio="none" aria-hidden="true"><path d={path} fill="none" stroke={color} strokeWidth="2" vectorEffect="non-scaling-stroke" /></svg>;
}

function KpiCard({ label, value, suffix, delta, color, icon: Icon, points, muted = false }: { label: string; value: string; suffix?: string; delta: string; color: string; icon: typeof Activity; points: number[]; muted?: boolean }) {
  return <div className={`card kpi-card ${muted ? 'kpi-muted' : ''}`} style={{ '--kpi-color': color } as React.CSSProperties}><div className="kpi-top"><span>{label}</span><Icon size={14} /></div><div className="kpi-number">{value}<small>{suffix}</small></div><div className="kpi-bottom"><span className="kpi-delta">{delta}</span><MiniSpark points={points} color={color} /></div></div>;
}
function PageHeader({ eyebrow, title, subtitle, action }: { eyebrow: string; title: string; subtitle: string; action?: ReactNode }) {
  return <div className="page-header"><div><div className="eyebrow">{eyebrow}</div><h1 className="page-title">{title}</h1><p className="page-subtitle">{subtitle}</p></div>{action}</div>;
}
function downloadLocal(filename: string, content: string, type = 'application/json') {
  const blob = new Blob([content], { type }); const url = URL.createObjectURL(blob); const anchor = document.createElement('a');
  anchor.href = url; anchor.download = filename; anchor.click(); URL.revokeObjectURL(url); toast.success('Export prepared locally');
}

function CommandPalette({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [query, setQuery] = useState('');
  const routes = [...navItems, ...opsItems];
  const filtered = routes.filter((item) => item.label.toLowerCase().includes(query.toLowerCase()));
  useEffect(() => { if (open) setQuery(''); }, [open]);
  if (!open) return null;
  return <div className="palette-backdrop" onClick={onClose}><div className="command-palette" onClick={(event) => event.stopPropagation()}><div className="palette-search"><Search size={15} /><input autoFocus value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search views and actions…" /><kbd>ESC</kbd></div><div className="palette-label">Navigate</div>{filtered.map(({ href, label, icon: Icon }) => <Link href={href} className="palette-item" key={href} onClick={onClose}><Icon size={15} /><span>{label}</span><ChevronRight size={13} className="ml-auto" /></Link>)}<div className="palette-hint"><Command size={12} /> Ctrl/Cmd + K to open · Esc to close</div></div></div>;
}

function Shell({ children }: { children: ReactNode }) {
  const [location] = useLocation(); const [open, setOpen] = useState(false); const [paletteOpen, setPaletteOpen] = useState(false);
  const [range, setRange] = useState('24H'); const [refresh, setRefresh] = useState(true); const { theme, toggleTheme } = useContext(ThemeContext);
  
  const { data: overview } = useQuery({ queryKey: ['overview'], queryFn: api.getOverview, enabled: refresh, refetchInterval: 10000 });
  const statusInfo = overview?.status || { snapshot: '—', tx: 0, wallets: 0, alerts: 0 };

  return <div className={`app-shell theme-${theme}`}><aside className={`sidebar ${open ? 'open' : ''}`}><div className="brand"><Link href="/overview" className="brand-link"><span className="brand-mark">CS</span><div><div className="brand-name">ChainSentinel</div><div className="brand-sub">offline forensics / v2.4</div></div></Link><button className="btn btn-quiet mobile-menu" onClick={() => setOpen(!open)} aria-label="Toggle navigation">{open ? <X size={17} /> : <Menu size={17} />}</button></div><nav className="nav-group"><div className="nav-label">Command center</div>{navItems.map(({ href, label, icon: Icon }) => <Link key={href} href={href} className={`nav-item ${location === href ? 'active' : ''}`} onClick={() => setOpen(false)}><Icon size={15} strokeWidth={1.7} /><span>{label}</span>{label === 'Alerts' && statusInfo.alerts > 0 && <span className="nav-count">{statusInfo.alerts}</span>}</Link>)}<div className="nav-label nav-label-spaced">Operations</div>{opsItems.map(({ href, label, icon: Icon }) => <Link key={href} href={href} className={`nav-item ${location === href ? 'active' : ''}`} onClick={() => setOpen(false)}><Icon size={15} strokeWidth={1.7} /><span>{label}</span></Link>)}</nav><div className="sidebar-footer"><div className="sidebar-status"><span className="live-dot" />Local snapshot mounted</div><div className="sidebar-meta">{statusInfo.snapshot}<br />{statusInfo.tx} tx · {statusInfo.wallets} wallets</div></div></aside><div className="main-area"><header className="topbar"><div className="topbar-left"><button className="global-search" onClick={() => setPaletteOpen(true)}><Search size={14} /><span>Search or jump to…</span><kbd>⌘K</kbd></button><div className="topbar-crumb">/ {[...navItems, ...opsItems].find((item) => item.href === location)?.label ?? 'Overview'}</div></div><div className="topbar-actions"><div className="range-picker">{['1H', '24H', '7D'].map((item) => <button key={item} className={range === item ? 'selected' : ''} onClick={() => setRange(item)}>{item}</button>)}</div><button className={`auto-refresh ${refresh ? 'on' : ''}`} onClick={() => setRefresh(!refresh)} title="Toggle auto-refresh"><RefreshCw size={13} className={refresh ? 'spin-slow' : ''} />{refresh ? 'LIVE' : 'PAUSED'}</button><div className="threat-gauge"><span className="live-dot" />THREAT <b>HIGH</b></div><button className="theme-toggle" type="button" onClick={toggleTheme}><Sun size={14} /><span>Toggle theme</span></button><span className="analyst-avatar">MA</span></div></header><div className="monitoring-strip"><span><i className="live-dot" />system nominal</span><span>snapshot <b>{statusInfo.snapshot}</b></span><span>ingest <b>{statusInfo.tx} tx</b></span><span>open alerts <b>{statusInfo.alerts}</b></span><span>risk model <b>v2.4</b></span><span>mode <b>offline</b></span></div>{children}</div><CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} /></div>;
}


function OverviewPage() {
  return <Shell><main className="page overview-page"><PageHeader eyebrow="Command center / executive wall" title="Investigation overview" subtitle="Offline Bitcoin flow telemetry across the mounted snapshot." action={<div className="header-actions"><Link href="/alerts" className="btn btn-primary"><ShieldAlert size={14} />Review alerts</Link><button className="btn" onClick={() => downloadLocal('chainsentinel-overview.json', JSON.stringify({ telemetryData, alerts }, null, 2))}><Download size={14} />Export wall</button></div>} /><div className="kpi-wall">
    <KpiCard label="Total transactions ingested" value="450" suffix="K" delta="+18.4% vs prior" color="#f472d0" icon={Activity} points={[12, 18, 14, 24, 22, 31, 28]} />
    <KpiCard label="Network observations" value="200" suffix="K" delta="+9.1% correlated" color="#2fb8ff" icon={Radio} points={[20, 16, 21, 24, 18, 29, 32]} />
    <KpiCard label="Flagged entities" value="639" delta="+32 in range" color="#6bff6b" icon={ShieldAlert} points={[18, 20, 19, 24, 27, 25, 31]} />
    <KpiCard label="Critical alerts" value="04" delta="2 need review" color="#f472d0" icon={AlertTriangle} points={[2, 3, 2, 5, 4, 7, 4]} />
    <KpiCard label="IP ↔ wallet links" value="3.08" suffix="K" delta="+12.6% confidence" color="#ffe23a" icon={ArrowLeftRight} points={[12, 11, 15, 14, 18, 17, 21]} />
    <KpiCard label="Illicit volume traced" value="244" suffix="K" delta="87.2 BTC tainted" color="#2fb8ff" icon={WalletCards} points={[15, 18, 17, 23, 21, 28, 26]} />
    <KpiCard label="Wallet clusters found" value="15" delta="+3 since run" color="#8b5cf6" icon={Boxes} points={[3, 4, 4, 7, 8, 10, 12]} />
    <KpiCard label="Average risk score" value="63.8" delta="↑ 4.2 points" color="#ffe23a" icon={BarChart3} points={[39, 44, 43, 49, 54, 58, 64]} />
    <KpiCard label="Model confidence" value="88.4" suffix="%" delta="calibrated / stable" color="#6bff6b" icon={Cpu} points={[74, 77, 76, 81, 84, 86, 88]} />
    <KpiCard label="Mixer / CoinJoin hits" value="31" delta="7 high confidence" color="#f472d0" icon={GitMerge} points={[12, 13, 18, 19, 21, 26, 31]} />
    <KpiCard label="Cash-out exposure" value="18.7" suffix="BTC" delta="across 4 exchanges" color="#ef4f79" icon={ArrowUpRight} points={[7, 9, 8, 12, 11, 16, 19]} />
    <KpiCard label="Pipeline health / uptime" value="No data" delta="local probe unavailable" color="#6b7280" icon={CircleAlert} points={[2, 2, 2, 2, 2, 2, 2]} muted />
  </div>
  <Panel title="Traffic & flow velocity" note="Transactions and network observations per 15 minutes · anomaly spikes are selectable" className="telemetry-panel section-gap" action={<div className="panel-actions"><span className="live-label"><span className="live-dot" />streaming</span><span className="tag">5,004 tx</span></div>}><div className="telemetry-chart"><ResponsiveContainer width="100%" height="100%"><AreaChart data={telemetryData} margin={{ top: 12, right: 17, bottom: 0, left: -14 }}><defs><linearGradient id="tx-fill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor="#2fb8ff" stopOpacity=".34" /><stop offset="100%" stopColor="#2fb8ff" stopOpacity="0" /></linearGradient><linearGradient id="network-fill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor="#f472d0" stopOpacity=".28" /><stop offset="100%" stopColor="#f472d0" stopOpacity="0" /></linearGradient></defs><CartesianGrid vertical={false} stroke="rgba(130,150,190,.18)" strokeDasharray="2 5" /><XAxis dataKey="time" tick={{ fontSize: 9 }} axisLine={false} tickLine={false} /><YAxis tick={{ fontSize: 9 }} axisLine={false} tickLine={false} /><Tooltip contentStyle={{ fontSize: 10, border: '1px solid rgba(130,150,190,.3)', borderRadius: 3, background: '#11111c', color: '#ecebf5' }} /><Legend verticalAlign="top" align="right" height={26} iconType="plainline" wrapperStyle={{ fontSize: 9, fontFamily: 'IBM Plex Mono' }} /><Area type="monotone" dataKey="tx" name="tx volume" stroke="#2fb8ff" fill="url(#tx-fill)" strokeWidth={2} /><Area type="monotone" dataKey="network" name="network events" stroke="#f472d0" fill="url(#network-fill)" strokeWidth={1.8} /><Line type="monotone" dataKey="flagged" name="flagged" stroke="#ffe23a" strokeWidth={1.3} dot={{ r: 2, fill: '#ffe23a' }} /></AreaChart></ResponsiveContainer></div><div className="anomaly-ruler"><span>00:00</span><span>06:00</span><span className="anomaly-marker"><i />12:00 anomaly spike · ALT-184</span><span>18:00</span><span>24:00</span></div></Panel>
  <div className="insight-grid section-gap"><Panel title="Risk distribution" note="176 monitored wallets"><div className="compact-chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={riskDistribution} margin={{ top: 16, right: 10, bottom: 0, left: -24 }}><CartesianGrid vertical={false} stroke="rgba(130,150,190,.14)" /><XAxis dataKey="name" tick={{ fontSize: 8 }} axisLine={false} tickLine={false} /><YAxis tick={{ fontSize: 8 }} axisLine={false} tickLine={false} /><Bar dataKey="value" radius={[2, 2, 0, 0]}>{riskDistribution.map((entry) => <Cell key={entry.name} fill={entry.color} />)} </Bar></BarChart></ResponsiveContainer></div><div className="panel-footer-stat"><span><i className="swatch critical" />Critical + high</span><b>26 / 176</b></div></Panel>
    <Panel title="Alerts by typology" note="Ranked in current snapshot"><div className="compact-chart"><ResponsiveContainer width="100%" height="100%"><PieChart><Pie data={typologyData} dataKey="value" nameKey="name" cx="50%" cy="48%" innerRadius={45} outerRadius={73} paddingAngle={3} stroke="transparent">{typologyData.map((entry) => <Cell key={entry.name} fill={entry.color} />)}</Pie><Tooltip contentStyle={{ fontSize: 10, border: 0, background: '#11111c' }} /></PieChart></ResponsiveContainer></div><div className="donut-legend">{typologyData.slice(0, 4).map((item) => <span key={item.name}><i style={{ background: item.color }} />{item.name} <b>{item.value}</b></span>)}</div></Panel>
    <Panel title="Top countries / ASNs" note="Observed network enrichment"><div className="bar-list">{geoData.map((item) => <div className="bar-row" key={item.country}><span>{item.country}</span><div><i style={{ width: `${item.value / 40 * 100}%`, background: item.color }} /></div><b>{item.value}</b></div>)}</div><div className="panel-footer-stat"><span>Top ASN</span><b className="mono">AS9009 · M247</b></div></Panel>
    <Panel title="Live alert ticker" note="Most recent high-signal events" action={<Link href="/alerts" className="btn btn-quiet">Open queue <ChevronRight size={12} /></Link>}><div className="ticker-list">{alerts.slice(0, 4).map((alert) => <Link href="/alerts" className="ticker-item" key={alert.id}><span className={`ticker-dot ${riskLevel(alert.risk_score)}`} /><div><strong>{alert.id} · {alert.matched_pattern}</strong><span>{alert.wallet_id} · {alert.lastSeen}</span></div><RiskBadge score={alert.risk_score} compact /></Link>)}</div></Panel>
    <Panel title="Threat heatmap" note="Alert intensity by hour / day"><div className="heatmap"><div className="heatmap-days">{['M','T','W','T','F','S','S'].map((day, index) => <span key={`${day}-${index}`}>{day}</span>)}</div>{Array.from({ length: 42 }, (_, index) => <i key={index} className={`heat-${(index * 7 + index % 5) % 5}`} title={`signal intensity ${(index * 7 + index) % 100}`} />)}</div><div className="heatmap-scale"><span>low</span><i className="heat-0" /><i className="heat-2" /><i className="heat-4" /><span>high</span></div></Panel>
  </div>
  </main></Shell>;
}

function AlertDrawer({ alert, onClose }: { alert: Alert; onClose: () => void }) {
  return <aside className="alert-drawer card"><div className="drawer-head"><div><div className="eyebrow">Alert detail / {alert.id}</div><h2>{alert.matched_pattern}</h2></div><button className="btn btn-quiet" onClick={onClose} aria-label="Close alert detail"><X size={15} /></button></div><div className="drawer-risk"><div className={`risk-gauge risk-${riskLevel(alert.risk_score)}`}><strong>{alert.risk_score}</strong><span>risk / 100</span></div><div className="confidence-ring"><strong>{alert.confidence}%</strong><span>confidence</span></div><StatusPill status={alert.status} /></div><div className="drawer-section"><div className="eyebrow">Why flagged</div>{alert.contributing_signals.map((signal) => <div className="signal" key={signal}><span>{signal}</span><b>+ signal</b></div>)}</div><div className="drawer-section"><div className="eyebrow">Evidence timeline</div><ol className="mini-timeline"><li><b>{alert.firstSeen}</b><span>network observation matched to wallet candidate</span></li><li><b>18 May 13:51</b><span>blockchain event increased propagation confidence</span></li><li><b>{alert.lastSeen}</b><span>risk score crossed analyst threshold</span></li></ol></div><div className="drawer-section"><div className="eyebrow">Neighborhood</div><div className="mini-neighborhood"><span>{alert.wallet_id}</span><i /><span>{alert.entity}</span><i /><span>mixer-17</span></div></div><div className="drawer-actions"><button className="btn btn-primary" onClick={() => toast.success(`${alert.id} added to case`)}><FileSearch size={13} />Add to case</button><button className="btn" onClick={() => toast.success(`${alert.id} escalated locally`)}><ArrowUpRight size={13} />Escalate</button><button className="btn btn-quiet" onClick={() => toast('Feedback recorded as false positive')}><Eye size={13} />False positive</button></div></aside>;
}

function AlertsPage() {
  const [selectedTxid, setSelectedTxid] = useState<string | null>(null);
  const [minRisk, setMinRisk] = useState(0.45);
  const queryClient = useQueryClient();

  const { data: alertsData, isLoading } = useQuery({ queryKey: ['alerts', minRisk], queryFn: () => api.getAlerts(minRisk) });
  const { data: details } = useQuery({ queryKey: ['alertDetail', selectedTxid], queryFn: () => api.getAlertDetail(selectedTxid!), enabled: !!selectedTxid });

  const statusMutation = useMutation({
    mutationFn: ({ id, status }: { id: string; status: string }) => api.updateAlertStatus(id, status),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['alerts'] });
      if (selectedTxid) queryClient.invalidateQueries({ queryKey: ['alertDetail', selectedTxid] });
      toast.success(`Alert matrix status updated on engine: ${variables.status}`);
    }
  });

  const activeItems = alertsData?.items || [];

  return <Shell><main className="page"><PageHeader eyebrow="Command center / alert queue" title="Ranked alert investigations" subtitle="Explainable transaction anomalies evaluated from structural network features." /><div className={`alert-workspace ${details ? 'has-drawer' : ''}`}><div className="card alert-table-card"><div className="filter-row"><select value={minRisk} onChange={(e) => setMinRisk(Number(e.target.value))} style={{ height: '28px', padding: '0 8px', borderRadius: '3px', border: '1px solid rgba(155,158,208,.2)', background: 'rgba(24,23,39,.84)', color: '#9a98b5', font: '9px var(--app-font-mono)' }}><option value={0.45}>Medium Risk (0.45+)</option><option value={0.65}>High Risk (0.65+)</option><option value={0.85}>Critical Matrix (0.85+)</option></select></div><div className="table-wrap"><table className="data-table dense-table"><thead><tr><th>ID</th><th>Cluster</th><th>Wallet Output</th><th>Typology Pattern</th><th>Risk Score</th><th>Country</th><th>Status</th></tr></thead><tbody>{activeItems.map((alert: any) => <tr key={alert.txid} onClick={() => setSelectedTxid(alert.txid)} className={selectedTxid === alert.txid ? 'selected-row' : ''}><td><strong className="table-primary">{alert.id}</strong></td><td><span className="mono">{alert.entity}</span></td><td><span className="mono table-secondary">{alert.wallet_full ? (alert.wallet_full.length > 12 ? alert.wallet_full.substring(0,6) + '…' + alert.wallet_full.substring(alert.wallet_full.length - 4) : alert.wallet_full) : alert.wallet_id}</span></td><td><span className="typology-tag">{alert.typology}</span></td><td><RiskBadge score={alert.risk_score} compact /></td><td><span className="mono">{alert.country}</span></td><td><StatusPill status={alert.status} /></td></tr>)}</tbody></table></div></div>
  {details && (
    <aside className="alert-drawer card"><div className="drawer-head"><div><div className="eyebrow">Alert details / {details.id}</div><h2>{details.matched_pattern}</h2></div><button className="btn btn-quiet" onClick={() => setSelectedTxid(null)}><X size={15} /></button></div><div className="drawer-risk"><div className="risk-gauge"><strong>{details.risk_score}</strong><span>risk weight</span></div><div className="confidence-ring"><strong>{details.confidence}%</strong><span>confidence</span></div><StatusPill status={details.status} /></div><div className="drawer-section"><div className="eyebrow">Signals matched</div>{details.contributing_signals?.map((sig: string) => <div className="signal" key={sig}><span>{sig}</span></div>)}</div>
      <div className="drawer-section"><div className="eyebrow">SHAP Feature Contributions</div>{details.shap_features?.map((feat: any) => <div className="signal" key={feat.feature}><span>{feat.feature}</span><b className={Number(feat.value) < 0 ? 'text-low' : 'text-critical'}>{feat.value}</b></div>)}</div>
      <div className="drawer-actions">
        <button className="btn btn-primary" onClick={() => statusMutation.mutate({ id: details.txid, status: 'Investigating' })}>Assign Review</button>
        <button className="btn" onClick={() => statusMutation.mutate({ id: details.txid, status: 'False positive' })}>Dismiss Variant</button>
      </div></aside>
  )}</div></main></Shell>;
}


type GraphNode = { id: string; label: string; x: number; y: number; score: number; type: 'wallet' | 'entity' | 'service' | 'ip' };
const graphNodes: GraphNode[] = [
  { id: 'bc1q8m…7k2p', label: 'bc1q8m…7k2p', x: 140, y: 195, score: 92, type: 'wallet' }, { id: 'ent-0042', label: 'ent-0042', x: 330, y: 125, score: 82, type: 'entity' }, { id: '3FZbgi…Qx4r', label: '3FZbgi…Qx4r', x: 520, y: 200, score: 81, type: 'wallet' }, { id: 'mixer-17', label: 'mixer-17', x: 725, y: 120, score: 91, type: 'service' }, { id: 'bc1q3a…v9d1', label: 'bc1q3a…v9d1', x: 180, y: 400, score: 68, type: 'wallet' }, { id: 'ent-0018', label: 'ent-0018', x: 400, y: 355, score: 63, type: 'entity' }, { id: 'exchange-03', label: 'exchange-03', x: 680, y: 390, score: 49, type: 'service' }, { id: '10.48.2.17', label: '10.48.2.17', x: 590, y: 480, score: 56, type: 'ip' }, { id: 'bc1q7p…2me8', label: 'bc1q7p…2me8', x: 790, y: 510, score: 47, type: 'wallet' },
];
const graphEdges = [['bc1q8m…7k2p', 'ent-0042'], ['ent-0042', '3FZbgi…Qx4r'], ['3FZbgi…Qx4r', 'mixer-17'], ['bc1q3a…v9d1', 'ent-0018'], ['ent-0018', 'exchange-03'], ['ent-0018', 'ent-0042'], ['exchange-03', '10.48.2.17'], ['10.48.2.17', 'bc1q7p…2me8']];
function GraphPage() {
  const [minRisk, setMinRisk] = useState(0.45);
  const [query, setQuery] = useState('');

  const { data: graphData, isLoading } = useQuery({ queryKey: ['graph', query, minRisk], queryFn: () => api.getGraph(query, minRisk) });

  const nodes = graphData?.nodes || [];
  const edges = graphData?.edges || [];

  return <Shell><main className="page"><PageHeader eyebrow="Command center / graph explorer" title="Entity relationship graph" subtitle="Multi-hop layout coordinates projected natively from GraphSAGE dimensions." /><div className="graph-layout"><div className="card graph-card"><div className="graph-toolbar"><div className="searchbox"><Search size={13} /><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search hash address, cluster, or IP..." /></div><select value={minRisk} onChange={(e) => setMinRisk(Number(e.target.value))} style={{ height: '28px', padding: '0 8px', borderRadius: '3px', border: '1px solid rgba(155,158,208,.2)', background: 'rgba(24,23,39,.84)', color: '#9a98b5', font: '9px var(--app-font-mono)', marginLeft: '10px' }}><option value={0.45}>Risk Threshold (0.45+)</option><option value={0.65}>Risk Threshold (0.65+)</option><option value={0.85}>Risk Threshold (0.85+)</option></select></div><div className="graph-canvas relationship-canvas"><svg className="graph-svg" viewBox="0 0 860 570">
    <defs><pattern id="graph-grid" width="32" height="32" patternUnits="userSpaceOnUse"><path d="M 32 0 L 0 0 0 32" fill="none" stroke="#2fb8ff" strokeWidth=".45" opacity=".2" /></pattern></defs><rect width="860" height="570" fill="url(#graph-grid)" />
    {edges.map((edge: string[], idx: number) => {
      const a = nodes.find((n: any) => n.id === edge[0]);
      const b = nodes.find((n: any) => n.id === edge[1]);
      if (!a || !b) return null;
      return <path key={idx} d={`M ${a.x} ${a.y} L ${b.x} ${b.y}`} fill="none" stroke="#415a72" strokeWidth={1.5} opacity={0.7} />;
    })}
    {nodes.map((node: any) => (
      <g key={node.id} transform={`translate(${node.x}, ${node.y})`}>
        <circle r={node.type === 'tx' ? 11 : 14} fill={node.type === 'ip' ? '#2fb8ff' : node.score >= 70 ? '#ef4f79' : '#34D399'} stroke="#fff" strokeWidth={1.5} />
        <text y={25} textAnchor="middle" fill="#aab9cc" className="mono" style={{ fontSize: '8px', paintOrder: 'stroke', stroke: '#07070d', strokeWidth: '3px', strokeLinejoin: 'round' }}>{node.label}</text>
      </g>
    ))}
  </svg></div></div></div></main></Shell>;
}


function TracerPage() {
  const [direction, setDirection] = useState<'forward' | 'backward'>('forward'); const [hops, setHops] = useState(4);
  const path = direction === 'forward' ? ['bc1q8m…7k2p', 'ent-0042', '3FZbgi…Qx4r', 'mixer-17', 'exchange-03'] : ['exchange-03', 'mixer-17', '3FZbgi…Qx4r', 'ent-0042', 'bc1q8m…7k2p'];
  return <Shell><main className="page"><PageHeader eyebrow="Command center / hop tracer" title="Hop-to-hop tracer" subtitle="Follow the money forward or inspect the source of funds with taint propagation." action={<button className="btn btn-primary" onClick={() => downloadLocal('chainsentinel-trace.json', JSON.stringify({ path, direction, hops }, null, 2))}><Download size={13} />Export evidence</button>} /><div className="trace-controls card"><div className="trace-field"><label>Source wallet / TXID</label><div className="searchbox"><Search size={13} /><input defaultValue="bc1q8m…7k2p" /></div></div><div className="trace-field"><label>Direction</label><div className="segmented"><button className={direction === 'forward' ? 'active' : ''} onClick={() => setDirection('forward')}><ArrowUpRight size={13} />Follow the money</button><button className={direction === 'backward' ? 'active' : ''} onClick={() => setDirection('backward')}><ArrowDownLeft size={13} />Source of funds</button></div></div><div className="trace-field hop-control"><label>Maximum hops <b>{hops}</b></label><input type="range" min="1" max="8" value={hops} onChange={(event) => setHops(Number(event.target.value))} /></div><button className="btn btn-primary trace-run" onClick={() => toast.success('Trace expanded across local snapshot')}><Zap size={13} />Run trace</button></div><div className="trace-summary section-gap"><div className="card"><span>tainted volume</span><strong>2.40 BTC</strong><small>poison model / retained 61%</small></div><div className="card"><span>path confidence</span><strong>88.2%</strong><small>3 observed · 1 inferred edge</small></div><div className="card"><span>pattern match</span><strong className="text-critical">peel + mixer</strong><small>high utility typology</small></div><div className="card"><span>elapsed</span><strong>31h 08m</strong><small>first to last seen</small></div></div><Panel title="Propagation path" note={`${path.length - 1} hops · inferred edges are dashed`} action={<span className="tag tag-critical">taint retained</span>} className="section-gap"><div className="trace-path">{path.map((node, index) => <Fragment key={node}><div className={`hop-card ${index === 0 ? 'source' : index === path.length - 1 ? 'terminal' : ''}`}><div className="hop-index">HOP {String(index).padStart(2, '0')}</div><strong>{node}</strong><span>{index === 0 ? 'source wallet' : index === path.length - 1 ? 'cash-out endpoint' : 'observed transfer'}</span><div className="hop-meta"><b>{(2.4 / (index + 1)).toFixed(2)} BTC</b><span>fee 0.0003</span><span>+{index * 7 + 4}h</span></div><div className="taint-meter"><i style={{ width: `${Math.max(22, 100 - index * 17)}%` }} /></div><small>taint {Math.max(22, 100 - index * 17)}%</small></div>{index < path.length - 1 && <div className="hop-arrow"><ArrowLeftRight size={15} /><span>{index % 2 ? 'inferred' : 'observed'}</span></div>}</Fragment>)}</div></Panel><div className="two-col section-gap"><Panel title="Pattern signatures" note="Matched behaviors along this path"><div className="signature-list">{['Peel chain continuation', 'Rapid pass-through', 'Mixer entry', 'Exchange cash-out'].map((item, index) => <div key={item}><CircleCheck size={14} /><span>{item}</span><b>{[96, 81, 89, 64][index]}%</b></div>)}</div></Panel><Panel title="Transaction evidence" note="Local block events"><div className="table-wrap"><table className="data-table"><thead><tr><th>TXID</th><th>Amount</th><th>Script</th><th>Timestamp</th></tr></thead><tbody>{transactions.map((tx) => <tr key={tx.txid}><td className="mono">{tx.txid}</td><td className="mono">{tx.amounts[0]} BTC</td><td>p2wpkh</td><td className="mono muted">{tx.timestamp}</td></tr>)}</tbody></table></div></Panel></div></main></Shell>;
}

function ClustersPage() {
  return <Shell><main className="page"><PageHeader eyebrow="Intelligence / entities" title="Entities & clusters" subtitle="Wallet groupings, behavior fingerprints, and cross-cluster value flow." action={<button className="btn btn-primary" onClick={() => toast.success('Cluster model recalculated locally')}><RefreshCw size={13} />Recalculate</button>} /><div className="cluster-grid">{entities.map((entity) => <div className="card cluster-card" key={entity.id}><div className="cluster-method">{entity.method}</div><div className="cluster-card-top"><div className="cluster-id">{entity.id}</div><RiskBadge score={entity.risk} compact /></div><div className="cluster-stats"><div><small>Wallets</small><strong>{entity.wallet_count}</strong></div><div><small>Volume</small><strong>{entity.total_volume.toFixed(1)} BTC</strong></div><div><small>Linked IPs</small><strong>{entity.ips}</strong></div></div><div className="cluster-tags"><span>{entity.typology}</span><span>{entity.country}</span></div><Link href="/graph" className="cluster-link">Inspect relationship graph <ChevronRight size={12} /></Link></div>)}</div><div className="two-col section-gap"><Panel title="Entity relationship matrix" note="Relative confidence between top clusters"><div className="matrix">{entities.slice(0, 5).map((row, r) => <div className="matrix-row" key={row.id}><span>{row.id}</span>{entities.slice(0, 5).map((col, c) => <i key={col.id} className={`matrix-${(r * 3 + c * 2) % 5}`} title={`${row.id} × ${col.id}`} />)}</div>)}</div></Panel><Panel title="Behavior fingerprint" note="Selected entity / ent-0042"><div className="fingerprint"><div className="radar-placeholder"><span>velocity</span><span>fan-in</span><span>fan-out</span><span>fee</span><span>script mix</span><svg viewBox="0 0 220 150"><polygon points="110,10 198,75 165,137 55,137 22,75" fill="none" stroke="rgba(47,184,255,.35)" /><polygon points="110,31 168,74 147,113 69,113 48,74" fill="rgba(244,114,208,.13)" stroke="#f472d0" /></svg></div><div className="fingerprint-legend">{['velocity 84', 'fan-out 71', 'fee anomaly 58', 'night ratio 62'].map((item) => <div key={item}><i className="legend-dot" />{item}</div>)}</div></div></Panel></div></main></Shell>;
}

function TypologiesPage() {
  const [group, setGroup] = useState('typology'); const [watchlist, setWatchlist] = useState<string[]>([]);
  const typologies = [{ name: 'Ransomware collection', count: 12, volume: '48.2 BTC', risk: 96 }, { name: 'Darknet market', count: 18, volume: '31.7 BTC', risk: 88 }, { name: 'Extortion / sextortion', count: 9, volume: '14.9 BTC', risk: 79 }, { name: 'Mixer / tumbler', count: 31, volume: '87.2 BTC', risk: 92 }, { name: 'Layering / structuring', count: 24, volume: '54.4 BTC', risk: 74 }, { name: 'Exchange cash-out', count: 16, volume: '18.7 BTC', risk: 68 }];
  return <Shell><main className="page"><PageHeader eyebrow="Intelligence / typologies" title="Fraud typology search" subtitle="Group, filter, and watch patterns without leaving the local snapshot." action={<div className="header-actions"><button className="btn" onClick={() => toast.success('Watchlist exported locally')}><Download size={13} />Export watchlist</button><button className="btn btn-primary" onClick={() => toast.success('Natural-language query parsed')}><Sparkles size={13} />Parse query</button></div>} /><div className="typology-search card"><div className="searchbox wide"><Search size={14} /><input defaultValue="show wallets with >10 hops and mixer exposure" placeholder="Describe a pattern in plain language…" /><button className="btn btn-primary">Search</button></div><div className="facet-row"><span>Facets</span>{['ransomware', 'darknet market', 'mixer', 'layering', 'exchange cash-out', 'extortion'].map((facet) => <button className="facet-chip" key={facet}>{facet}<X size={11} /></button>)}</div></div><div className="view-controls section-gap"><div><span className="eyebrow">GROUP RESULTS BY</span><div className="segmented">{['typology', 'cluster', 'country', 'ASN', 'script type'].map((item) => <button key={item} className={group === item ? 'active' : ''} onClick={() => setGroup(item)}>{item}</button>)}</div></div><div className="watchlist-count"><WalletCards size={14} />watchlist <b>{watchlist.length}</b></div></div><div className="typology-grid">{typologies.map((item) => <div className="card typology-card" key={item.name}><div className="typology-card-head"><div><div className="eyebrow">group / {group}</div><h2>{item.name}</h2></div><RiskBadge score={item.risk} compact /></div><div className="typology-card-stats"><div><small>entities</small><strong>{item.count}</strong></div><div><small>traced volume</small><strong>{item.volume}</strong></div><div><small>top country</small><strong>{['NL', 'RU', 'US', 'DE', 'GB', 'SG'][item.count % 6]}</strong></div></div><button className={`watch-btn ${watchlist.includes(item.name) ? 'added' : ''}`} onClick={() => setWatchlist((current) => current.includes(item.name) ? current.filter((name) => name !== item.name) : [...current, item.name])}>{watchlist.includes(item.name) ? <Check size={12} /> : <WalletCards size={12} />}{watchlist.includes(item.name) ? 'On watchlist' : 'Add to watchlist'}</button></div>)}</div></main></Shell>;
}

function CorrelationPage() {
  return <Shell><main className="page"><PageHeader eyebrow="Intelligence / network correlation" title="Network correlation" subtitle="Match IP observations to blockchain broadcasts with confidence-aware evidence." action={<button className="btn btn-primary" onClick={() => toast.success('Correlation export prepared')}><Download size={13} />Export matches</button>} /><Panel title="Correlation swim lanes" note="IP / port observations against transaction broadcasts · 18 May 2024"><div className="swimlane"><div className="lane-label"><Radio size={13} />IP / port observations</div><div className="lane-grid">{[12, 24, 36, 48, 60, 72, 84].map((tick) => <i key={tick} style={{ left: `${tick}%` }} />)}{[18, 31, 48, 52, 76].map((left, index) => <b key={left} style={{ left: `${left}%`, height: `${24 + index * 8}px` }} />)}</div><div className="lane-label"><Database size={13} />TX broadcasts</div><div className="lane-grid second">{[12, 24, 36, 48, 60, 72, 84].map((tick) => <i key={tick} style={{ left: `${tick}%` }} />)}{[22, 35, 50, 57, 80].map((left, index) => <b key={left} style={{ left: `${left}%`, height: `${19 + index * 8}px` }} />)}</div><svg className="correlation-lines" viewBox="0 0 900 170" preserveAspectRatio="none"><path d="M160 48 C250 130 310 40 385 128" /><path d="M305 45 C420 125 430 35 505 125" /><path d="M472 46 C590 135 620 35 720 125" /><path d="M550 46 C650 110 700 55 790 125" /></svg></div><div className="axis-labels"><span>00:00</span><span>06:00</span><span>12:00</span><span>18:00</span><span>24:00</span></div></Panel><div className="correlation-grid section-gap"><Panel title="Candidate matches" note="IP → wallet correlation candidates"><div className="table-wrap"><table className="data-table"><thead><tr><th>IP / ASN</th><th>Wallet</th><th>Confidence</th><th>Evidence</th><th>Geo</th></tr></thead><tbody>{[['10.48.2.17', 'bc1q8m…7k2p', '94%', '12 events', 'NL'], ['172.20.4.91', '3FZbgi…Qx4r', '87%', '8 events', 'RU'], ['192.168.44.8', 'bc1q3a…v9d1', '79%', '6 events', 'RU'], ['10.9.11.43', 'bc1q7p…2me8', '66%', '4 events', 'DE']].map((row) => <tr key={row[0]}>{row.map((cell, index) => <td className={index === 0 || index === 1 ? 'mono' : ''} key={cell}>{cell}</td>)}</tr>)}</tbody></table></div></Panel><Panel title="Confidence breakdown" note="Selected match / 10.48.2.17"><div className="confidence-bars">{[['Time proximity', 94], ['First-seen relay', 88], ['IP co-occurrence', 91], ['ASN consistency', 76]].map(([label, value]) => <div key={label as string}><span>{label}</span><b>{value}%</b><i><em style={{ width: `${value}%` }} /></i></div>)}</div><div className="correlation-score"><span>composite score</span><strong>0.89</strong></div></Panel></div></main></Shell>;
}

function GeoPage() {
  const countries = [{ name: 'Netherlands', code: 'NL', score: 92, alerts: 18, volume: '38.4 BTC' }, { name: 'Russian Federation', code: 'RU', score: 81, alerts: 14, volume: '31.7 BTC' }, { name: 'United States', code: 'US', score: 68, alerts: 11, volume: '27.3 BTC' }, { name: 'Germany', code: 'DE', score: 47, alerts: 8, volume: '18.1 BTC' }, { name: 'Singapore', code: 'SG', score: 35, alerts: 5, volume: '11.2 BTC' }];
  return <Shell><main className="page"><PageHeader eyebrow="Intelligence / geo map" title="Offline geo intelligence" subtitle="Country and ASN enrichment from the mounted local GeoIP dataset." action={<span className="tag tag-closed"><Check size={11} />local GeoJSON loaded</span>} /><div className="geo-layout"><Panel title="Risk by country" note="Arc routes show observed cross-border flow"><div className="offline-map"><div className="map-grid" /><div className="map-continent c1" /><div className="map-continent c2" /><div className="map-continent c3" /><svg viewBox="0 0 800 360"><path d="M155 112 C280 35 365 110 448 102 S590 75 682 124" /><path d="M222 235 C330 160 460 168 620 242" /><path d="M348 75 C385 140 410 220 512 292" /></svg>{countries.map((country, index) => <div className={`map-bubble bubble-${index}`} key={country.code}><b>{country.code}</b><span>{country.alerts}</span></div>)}</div><div className="map-legend"><span><i className="legend-line solid" />flow route</span><span><i className="legend-dot" />alert density</span><span className="mono">source: geoip2-local.mmdb</span></div></Panel><Panel title="Country drill-down" note="Sorted by composite risk"><div className="country-list">{countries.map((country) => <div className="country-row" key={country.code}><div className="country-code">{country.code}</div><div><strong>{country.name}</strong><span>{country.alerts} alerts · {country.volume}</span></div><RiskBadge score={country.score} compact /><ChevronRight size={13} /></div>)}</div></Panel></div><div className="card section-gap geo-footnote"><Globe2 size={14} /><span>Map tiles are not used. All geometry and enrichment are served from local assets so this workspace remains air-gapped.</span></div></main></Shell>;
}

function ExplainPage() {
  const [values, setValues] = useState([72, 61, 43, 38]); const liveScore = Math.min(99, Math.round(32 + values.reduce((sum, value) => sum + value, 0) / 6));
  const features = ['Mixer proximity', 'Fan-out ratio', 'Velocity anomaly', 'Fee anomaly', 'Address age', 'Night-time ratio'];
  return <Shell><main className="page"><PageHeader eyebrow="Intelligence / explainability studio" title="Explainability studio" subtitle="Inspect feature contributions, test counterfactuals, and generate an offline analyst brief." action={<button className="btn btn-primary" onClick={() => toast.success('Analyst brief regenerated from local template')}><RefreshCw size={13} />Regenerate brief</button>} /><div className="explain-layout"><div><Panel title="SHAP contribution / ALT-184" note="Selected alert · waterfall view"><div className="shap-chart">{[['Counterparty risk', 41, 'positive'], ['Velocity anomaly', 23, 'positive'], ['Address age', 17, 'positive'], ['Script mix', -9, 'negative'], ['Baseline', 30, 'base']].map(([label, value, kind]) => <div className="shap-row" key={label as string}><span>{label}</span><i><em className={kind as string} style={{ width: `${Math.abs(Number(value)) * 1.7}%` }} /></i><b className={kind as string}>{Number(value) > 0 ? '+' : ''}{value}</b></div>)}</div></Panel><Panel title="What-if score simulator" note="Move top features to recompute risk live" className="section-gap"><div className="what-if-score"><span>projected risk</span><strong>{liveScore}</strong><RiskBadge score={liveScore} /></div><div className="slider-list">{features.slice(0, 4).map((feature, index) => <label key={feature}><span>{feature}<b>{values[index]}</b></span><input type="range" min="0" max="100" value={values[index]} onChange={(event) => setValues((current) => current.map((value, itemIndex) => itemIndex === index ? Number(event.target.value) : value))} /></label>)}</div><div className="counterfactual"><CircleAlert size={14} /><span>This would not have been flagged if mixer proximity fell below <b>21</b> and velocity anomaly below <b>34</b>.</span></div></Panel></div><div><Panel title="Analyst brief" note="AI-generated, verify before action" action={<span className="tag">offline template</span>}><div className="brief"><div className="brief-highlight">ALT-184 indicates a high-confidence layered movement pattern with probable mixer ingress.</div><h3>Summary</h3><p>The selected wallet consolidates value from a linked entity before sending a retained remainder into a known mixer-adjacent service. Timing and address-age signals are consistent with a peel sequence.</p><h3>Key indicators</h3><ul>{alerts[0].contributing_signals.map((signal) => <li key={signal}><Check size={12} />{signal}</li>)}</ul><h3>Recommended actions</h3><ol><li>Review the 4-hop propagation path.</li><li>Compare the IP candidate against the local ASN history.</li><li>Pin the evidence bundle before escalation.</li></ol><div className="brief-caveat"><CircleAlert size={13} />Confidence is calibrated against synthetic snapshot data. Do not treat this as attribution.</div></div><div className="drawer-actions"><button className="btn" onClick={() => navigator.clipboard?.writeText('ALT-184 analyst brief copied')}><FileText size={13} />Copy brief</button><button className="btn"><Sparkles size={13} />Ask follow-up</button></div></Panel><Panel title="Model contribution" note="Current alert ensemble" className="section-gap"><div className="model-bars">{[['Isolation Forest', 31], ['Autoencoder', 24], ['Graph features', 29], ['Clustering', 16]].map(([label, value]) => <div key={label as string}><span>{label}</span><i><em style={{ width: `${value as number * 2.5}%` }} /></i><b>{value}%</b></div>)}</div></Panel></div></div></main></Shell>;
}

function CasesPage() {
  const columns: { title: string; cls: string; items: string[] }[] = [{ title: 'New', cls: 'new', items: ['CASE-2024-017 · ent-0042', 'CASE-2024-016 · ent-0113'] }, { title: 'Investigating', cls: 'investigating', items: ['CASE-2024-012 · ent-0018'] }, { title: 'Escalated', cls: 'escalated', items: ['CASE-2024-009 · mixer-17'] }, { title: 'Closed', cls: 'closed', items: ['CASE-2024-003 · ent-0091'] }];
  return <Shell><main className="page"><PageHeader eyebrow="Operations / cases" title="Investigation cases" subtitle="Bundle alerts, graph snapshots, notes, and evidence into reviewable records." action={<button className="btn btn-primary" onClick={() => toast.success('New case draft opened locally')}><FileSearch size={13} />New case</button>} /><div className="kanban">{columns.map((column) => <div className="kanban-column" key={column.title}><div className="kanban-head"><span className={`kanban-dot ${column.cls}`} />{column.title}<b>{column.items.length}</b></div>{column.items.map((item, index) => <div className="card case-card" key={item} onClick={() => toast(`Opened ${item}`)}><div className="case-card-top"><span className="mono">{item.split(' · ')[0]}</span><StatusPill status={column.title === 'New' ? 'New' : column.title} /></div><strong>{item.split(' · ')[1]} propagation review</strong><p>{index % 2 ? 'Potential fan-in with linked IP observations.' : 'Layered movement across entity neighborhood.'}</p><div className="case-card-foot"><span><Bell size={11} />{index + 2} alerts</span><span><Clock3 size={11} />{index + 1}h ago</span></div></div>)}<button className="kanban-add" onClick={() => toast('Case draft ready')}>+ Add case</button></div>)}</div><div className="two-col section-gap"><Panel title="Selected case timeline" note="CASE-2024-017 / entity ent-0042"><ol className="timeline"><li><div className="timeline-time">18 May · 13:51 UTC</div><div className="timeline-text">Alert ALT-184 created from mixer adjacency and rapid consolidation.</div></li><li><div className="timeline-time">18 May · 14:03 UTC</div><div className="timeline-text">Analyst assigned; evidence snapshot pinned to basket.</div></li><li><div className="timeline-time">18 May · 14:18 UTC</div><div className="timeline-text">Propagation path extended to mixer-17 via f4a7c2…0b91.</div></li></ol></Panel><Panel title="Evidence basket" note="Ready for report generation"><div className="basket-items"><div><Network size={13} />Graph snapshot <b>1</b></div><div><Sparkles size={13} />SHAP waterfall <b>1</b></div><div><FileText size={13} />Analyst notes <b>3</b></div></div><button className="btn btn-primary w-full justify-center mt-4" onClick={() => downloadLocal('case-2024-017-report.html', '<h1>ChainSentinel Case Report</h1><p>Offline evidence bundle for CASE-2024-017.</p>', 'text/html')}><Download size={13} />Generate investigation report</button></Panel></div></main></Shell>;
}

function PipelinePage() {
  const [rerun, setRerun] = useState(false); const stages = [['Ingest', '5,004 records mounted', 100], ['Parse', '12,481 transactions normalized', 100], ['Normalize', '302 wallets resolved', 100], ['Enrich', 'GeoIP / ASN local lookup', 100], ['Correlate', '3,084 candidates scored', 100], ['Graph build', '15 clusters / 7,420 edges', 100], ['ML + alerts', rerun ? 'recalculating local scores…' : '60 alerts ranked', rerun ? 54 : 100]];
  return <Shell><main className="page"><PageHeader eyebrow="Operations / pipeline" title="Pipeline & model" subtitle="Reproducible ingest stages, local model health, and threshold controls." action={<button className="btn btn-primary" onClick={() => { setRerun(true); window.setTimeout(() => setRerun(false), 1100); }}><RefreshCw size={13} className={rerun ? 'spin-slow' : ''} />{rerun ? 'Running locally…' : 'Run pipeline'}</button>} /><div className="two-col"><Panel title="Pipeline stages" note="No network calls are made by this workspace" action={<span className="tag tag-closed"><CircleCheck size={11} />healthy</span>}><div className="pipeline-stages">{stages.map(([title, detail, progress]) => <div className="pipeline-stage" key={title as string}><div className="pipeline-check">{Number(progress) === 100 ? <Check size={13} /> : <Timer size={13} />}</div><div><strong>{title}</strong><span>{detail}</span><i><em style={{ width: `${progress}%` }} /></i></div><b>{progress === 100 ? 'done' : 'run'}</b></div>)}</div></Panel><Panel title="Model card" note="Risk ensemble / v2.4"><div className="model-card"><div className="model-metric"><span>precision</span><strong>0.91</strong></div><div className="model-metric"><span>recall</span><strong>0.84</strong></div><div className="model-metric"><span>F1</span><strong>0.87</strong></div><div className="model-metric"><span>AUC</span><strong>0.93</strong></div></div><div className="threshold-control"><div><span>Alert threshold</span><b>0.70</b></div><input type="range" min="30" max="90" defaultValue="70" /><div className="threshold-scale"><span>more recall</span><span>fewer false positives</span></div></div><div className="model-note"><Cpu size={14} />Isolation Forest + graph features + clustering · last calibrated 18 May 2024</div></Panel></div><div className="card section-gap"><div className="card-head"><div><h2 className="section-title">Run metadata</h2><div className="section-note mt-1">Deterministic snapshot context</div></div><Database size={15} /></div><div className="metadata-grid">{[['Snapshot', '2024-05-18'], ['Block height', '842,119'], ['Parser version', '0.8.4'], ['Run duration', '00:02:41'], ['Input records', '12,481'], ['Output entities', '64'], ['GeoIP database', 'local / 2024.05'], ['LLM endpoint', 'not configured']].map(([label, value]) => <div key={label}><small>{label}</small><strong className="mono">{value}</strong></div>)}</div></div></main></Shell>;
}

function AuditPage() {
  const entries = [['14:18:02', 'M. Alvarez', 'ALERT_ESCALATED', 'ALT-184', 'a83f…91c2', 'b6d1…78ef'], ['14:03:44', 'M. Alvarez', 'EVIDENCE_PINNED', 'CASE-2024-017', '4f12…aa08', 'a83f…91c2'], ['13:51:18', 'system', 'ALERT_CREATED', 'ALT-184', '1d92…44b3', '4f12…aa08'], ['13:42:09', 'system', 'CORRELATION_UPDATED', '10.48.2.17', '7c21…a901', '1d92…44b3'], ['12:08:55', 'M. Alvarez', 'CASE_OPENED', 'CASE-2024-017', '08ff…cb31', '7c21…a901']];
  return <Shell><main className="page"><PageHeader eyebrow="Operations / audit log" title="Tamper-evident audit log" subtitle="Hash-chained analyst actions from the local investigation workspace." action={<button className="btn btn-primary" onClick={() => downloadLocal('chainsentinel-audit.json', JSON.stringify(entries, null, 2))}><Download size={13} />Export log</button>} /><div className="audit-banner"><ShieldAlert size={15} /><div><strong>Chain integrity verified</strong><span>Last checked 14:26:08 UTC · 60 entries · no breaks detected</span></div><span className="tag tag-closed">verified</span></div><div className="card section-gap"><div className="filter-row"><div className="searchbox"><Search size={13} /><input placeholder="Filter action, actor, or entity…" /></div><button className="btn"><Clock3 size={13} />Last 24 hours</button><button className="btn"><ListFilter size={13} />All actions</button></div><div className="table-wrap"><table className="data-table audit-table"><thead><tr><th>Time</th><th>Actor</th><th>Action</th><th>Object</th><th>Hash</th><th>Previous hash</th></tr></thead><tbody>{entries.map((entry) => <tr key={`${entry[0]}-${entry[2]}`}>{entry.map((cell, index) => <td key={cell} className={index > 3 ? 'mono muted' : index === 0 ? 'mono' : ''}>{index === 2 ? <span className="audit-action">{cell}</span> : cell}</td>)}</tr>)}</tbody></table></div></div></main></Shell>;
}

function SettingsPage() {
  const [threshold, setThreshold] = useState(70); const [dnd, setDnd] = useState(false); const [compact, setCompact] = useState(false);
  return <Shell><main className="page"><PageHeader eyebrow="Operations / settings" title="Workspace settings" subtitle="Tune analyst preferences and local model behavior. Changes stay in this browser session." action={<button className="btn btn-primary" onClick={() => toast.success('Settings saved locally')}><Check size={13} />Save changes</button>} /><div className="settings-grid"><Panel title="Analyst experience" note="Personal console preferences"><div className="setting-row"><div><strong>Do not disturb</strong><span>Suppress visual pulses for new critical alerts.</span></div><button className={`switch ${dnd ? 'on' : ''}`} onClick={() => setDnd(!dnd)} aria-label="Toggle do not disturb"><i /></button></div><div className="setting-row"><div><strong>Compact density</strong><span>Reduce spacing across tables and panels.</span></div><button className={`switch ${compact ? 'on' : ''}`} onClick={() => setCompact(!compact)} aria-label="Toggle compact density"><i /></button></div><div className="setting-row"><div><strong>Keyboard shortcuts</strong><span>J / K alerts · E escalate · G graph · ? help</span></div><span className="tag tag-closed">enabled</span></div></Panel><Panel title="Risk model threshold" note="Preview precision / recall trade-off"><div className="threshold-big"><strong>{threshold}</strong><span>/ 100</span><RiskBadge score={threshold} /></div><input type="range" min="30" max="95" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} className="wide-range" /><div className="threshold-scale"><span>more recall / 91 alerts</span><span>fewer false positives / 18 alerts</span></div><div className="setting-note"><SlidersHorizontal size={14} />Changing the threshold only affects the local preview until the next pipeline run.</div></Panel><Panel title="Offline data sources" note="Mounted assets and integrations"><div className="source-row"><Database size={14} /><div><strong>snapshot_2024-05-18.ndjson</strong><span>4.8 MB · loaded</span></div><StatusPill status="Ready" /></div><div className="source-row"><Globe2 size={14} /><div><strong>geoip2-local.mmdb</strong><span>312 MB · loaded</span></div><StatusPill status="Ready" /></div><div className="source-row"><Sparkles size={14} /><div><strong>Ollama / llama.cpp</strong><span>No endpoint configured · template fallback</span></div><StatusPill status="No data" /></div><button className="btn w-full justify-center mt-4"><Upload size={13} />Mount another local dataset</button></Panel><Panel title="Workspace information" note="Build and session details"><div className="metadata-grid settings-meta">{[['Build', 'ChainSentinel v2.4'], ['Storage', 'browser-local'], ['Network', 'air-gapped'], ['Analyst', 'M. Alvarez'], ['Session started', '14:26:08 UTC'], ['Theme', 'dark / light']].map(([label, value]) => <div key={label}><small>{label}</small><strong>{value}</strong></div>)}</div></Panel></div></main></Shell>;
}

function Router() {
  return <Switch><Route path="/" component={OverviewPage} /><Route path="/overview" component={OverviewPage} /><Route path="/alerts" component={AlertsPage} /><Route path="/graph" component={GraphPage} /><Route path="/tracer" component={TracerPage} /><Route path="/clusters" component={ClustersPage} /><Route path="/typologies" component={TypologiesPage} /><Route path="/patterns" component={TypologiesPage} /><Route path="/correlation" component={CorrelationPage} /><Route path="/geo" component={GeoPage} /><Route path="/explain" component={ExplainPage} /><Route path="/cases" component={CasesPage} /><Route path="/case" component={CasesPage} /><Route path="/pipeline" component={PipelinePage} /><Route path="/audit" component={AuditPage} /><Route path="/settings" component={SettingsPage} /><Route component={NotFound} /></Switch>;
}

function RoutedErrorBoundary({ children }: { children: ReactNode }) {
  const [location] = useLocation();
  return <ErrorBoundary resetKey={location}>{children}</ErrorBoundary>;
}

function App() {
  const [theme, setTheme] = useState<ThemeMode>(() => {
    const stored = typeof window !== 'undefined' ? window.localStorage.getItem('chainsentinel-theme') : null;
    if (stored === 'light' || stored === 'dark') return stored;
    return typeof window !== 'undefined' && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  });
  const toggleTheme = () => setTheme((current) => { const next = current === 'dark' ? 'light' : 'dark'; window.localStorage.setItem('chainsentinel-theme', next); return next; });
  useEffect(() => { document.documentElement.dataset.theme = theme; }, [theme]);
  return <QueryClientProvider client={queryClient}><TooltipProvider><ThemeContext.Provider value={{ theme, toggleTheme }}><WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}><RoutedErrorBoundary><Router /></RoutedErrorBoundary></WouterRouter><Toaster theme={theme} position="bottom-right" /></ThemeContext.Provider></TooltipProvider></QueryClientProvider>;
}

export default App;