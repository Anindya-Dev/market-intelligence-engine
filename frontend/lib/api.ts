import {
  DashboardData,
  DashboardQuery,
  DrawdownPoint,
  EquityPoint,
  FeatureImportancePoint,
  PriceWithSignalPoint,
  ProbabilityPoint,
  RollingAUCPoint,
  TradeDistributionPoint,
} from "@/types/dashboard";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

type QueryParams = Record<string, string | number | boolean>;

function toUrl(path: string, params: QueryParams): string {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => qs.set(k, String(v)));
  return `${API_BASE}${path}?${qs.toString()}`;
}

async function getJson<T>(path: string, params: QueryParams): Promise<T> {
  const response = await fetch(toUrl(path, params), {
    method: "GET",
    headers: { "Content-Type": "application/json" },
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(`API ${path} failed: ${response.status}`);
  }

  return (await response.json()) as T;
}

function makeMockPriceSeries(): PriceWithSignalPoint[] {
  const now = Date.now();
  const size = 120;
  let last = 100;

  return Array.from({ length: size }).map((_, i) => {
    const drift = (Math.random() - 0.5) * 1.4;
    const open = last;
    const close = open + drift;
    const high = Math.max(open, close) + Math.random() * 0.8;
    const low = Math.min(open, close) - Math.random() * 0.8;
    const volume = 20 + Math.random() * 90;
    last = close;

    return {
      timestamp: new Date(now - (size - i) * 5 * 60_000).toISOString(),
      open,
      high,
      low,
      close,
      volume,
      signal: i % 29 === 0 ? "long" : i % 41 === 0 ? "short" : null,
      maFast: null,
      maSlow: null,
    };
  });
}

function makeMockProbabilitySeries(): ProbabilityPoint[] {
  return makeMockPriceSeries().map((d) => {
    const p = 0.45 + Math.random() * 0.2;
    return {
      timestamp: d.timestamp,
      probability: Number(p.toFixed(4)),
      signalActive: p >= 0.55,
    };
  });
}

function makeMockRollingAUC(): RollingAUCPoint[] {
  return makeMockPriceSeries().map((d) => ({
    timestamp: d.timestamp,
    auc: Number((0.48 + Math.random() * 0.08).toFixed(4)),
  }));
}

function makeMockEquityCurve(): EquityPoint[] {
  let equity = 100_000;
  return makeMockPriceSeries().map((d) => {
    equity *= 1 + (Math.random() - 0.49) * 0.004;
    return { timestamp: d.timestamp, equity: Number(equity.toFixed(2)) };
  });
}

function makeMockDrawdown(): DrawdownPoint[] {
  return makeMockPriceSeries().map((d) => ({
    timestamp: d.timestamp,
    drawdown: Number((-(Math.random() * 0.18)).toFixed(4)),
  }));
}

function makeMockFeatureImportance(): FeatureImportancePoint[] {
  return [
    { feature: "volume_change", importance: 0.28 },
    { feature: "momentum_10", importance: 0.22 },
    { feature: "rsi", importance: 0.19 },
    { feature: "rolling_volatility", importance: 0.14 },
    { feature: "atr", importance: 0.1 },
    { feature: "ma_spread", importance: 0.07 },
  ];
}

function makeMockTradeDistribution(): TradeDistributionPoint[] {
  return [
    { bucket: "<-2%", count: 3 },
    { bucket: "-2%- -1%", count: 8 },
    { bucket: "-1%-0%", count: 18 },
    { bucket: "0%-1%", count: 21 },
    { bucket: "1%-2%", count: 11 },
    { bucket: ">2%", count: 4 },
  ];
}

type OhlcApiRow = {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

function timeframeToMs(timeframe: DashboardQuery["timeframe"]): number {
  if (timeframe === "5m") return 5 * 60_000;
  if (timeframe === "15m") return 15 * 60_000;
  if (timeframe === "30m") return 30 * 60_000;
  return 60 * 60_000;
}

function resampleCandles(
  rows: OhlcApiRow[],
  timeframe: DashboardQuery["timeframe"],
): OhlcApiRow[] {
  const bucketMs = timeframeToMs(timeframe);
  const sorted = [...rows].sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
  const acc = new Map<number, OhlcApiRow>();

  for (const row of sorted) {
    const ts = new Date(row.timestamp).getTime();
    const bucket = Math.floor(ts / bucketMs) * bucketMs;
    const existing = acc.get(bucket);

    if (!existing) {
      acc.set(bucket, {
        timestamp: new Date(bucket).toISOString(),
        open: Number(row.open),
        high: Number(row.high),
        low: Number(row.low),
        close: Number(row.close),
        volume: Number(row.volume),
      });
      continue;
    }

    existing.high = Math.max(existing.high, Number(row.high));
    existing.low = Math.min(existing.low, Number(row.low));
    existing.close = Number(row.close);
    existing.volume += Number(row.volume);
  }

  return Array.from(acc.values()).sort(
    (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime(),
  );
}

function toPriceWithSignalPoints(rows: OhlcApiRow[]): PriceWithSignalPoint[] {
  return rows.map((r) => ({
    timestamp: r.timestamp,
    open: Number(r.open),
    high: Number(r.high),
    low: Number(r.low),
    close: Number(r.close),
    volume: Number(r.volume),
    signal: null,
    maFast: null,
    maSlow: null,
  }));
}

export async function fetchPriceWithSignals(query: DashboardQuery): Promise<PriceWithSignalPoint[]> {
  try {
    return await getJson<PriceWithSignalPoint[]>("/chart/price-with-signals", { ...query });
  } catch {
    try {
      const rows = await getJson<OhlcApiRow[]>("/ohlc", { symbol: query.symbol, limit: 800 });
      const sampled = resampleCandles(rows, query.timeframe);
      if (sampled.length > 0) {
        return toPriceWithSignalPoints(sampled.slice(-500));
      }
    } catch {
      // fallback to mock only if both chart and ohlc APIs are unavailable
    }

    return makeMockPriceSeries();
  }
}

export async function fetchProbabilitySeries(query: DashboardQuery): Promise<ProbabilityPoint[]> {
  try {
    return await getJson<ProbabilityPoint[]>("/chart/probability-series", { ...query });
  } catch {
    return makeMockProbabilitySeries();
  }
}

export async function fetchRollingAUC(query: DashboardQuery): Promise<RollingAUCPoint[]> {
  try {
    return await getJson<RollingAUCPoint[]>("/chart/rolling-auc", { ...query });
  } catch {
    return makeMockRollingAUC();
  }
}

export async function fetchEquityCurve(query: DashboardQuery): Promise<EquityPoint[]> {
  try {
    return await getJson<EquityPoint[]>("/chart/equity-curve", { ...query });
  } catch {
    return makeMockEquityCurve();
  }
}

export async function fetchDrawdownSeries(query: DashboardQuery): Promise<DrawdownPoint[]> {
  try {
    return await getJson<DrawdownPoint[]>("/chart/drawdown", { ...query });
  } catch {
    return makeMockDrawdown();
  }
}

export async function fetchFeatureImportance(query: DashboardQuery): Promise<FeatureImportancePoint[]> {
  try {
    return await getJson<FeatureImportancePoint[]>("/chart/feature-importance", { ...query });
  } catch {
    return makeMockFeatureImportance();
  }
}

export async function fetchTradeDistribution(query: DashboardQuery): Promise<TradeDistributionPoint[]> {
  try {
    return await getJson<TradeDistributionPoint[]>("/chart/trade-distribution", { ...query });
  } catch {
    return makeMockTradeDistribution();
  }
}

export async function fetchDashboardData(query: DashboardQuery): Promise<DashboardData> {
  const [
    priceSeries,
    probabilitySeries,
    rollingAucSeries,
    equityCurve,
    drawdownSeries,
    featureImportance,
    tradeDistribution,
  ] = await Promise.all([
    fetchPriceWithSignals(query),
    fetchProbabilitySeries(query),
    fetchRollingAUC(query),
    fetchEquityCurve(query),
    fetchDrawdownSeries(query),
    fetchFeatureImportance(query),
    fetchTradeDistribution(query),
  ]);

  return {
    priceSeries,
    probabilitySeries,
    rollingAucSeries,
    equityCurve,
    drawdownSeries,
    featureImportance,
    tradeDistribution,
  };
}

export async function fetchBackendHealth(): Promise<{ status: string }> {
  const response = await fetch(`${API_BASE}/health`, {
    method: "GET",
    headers: { "Content-Type": "application/json" },
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(`API /health failed: ${response.status}`);
  }

  return (await response.json()) as { status: string };
}
