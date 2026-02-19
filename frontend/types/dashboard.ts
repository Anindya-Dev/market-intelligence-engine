export type Timeframe = "5m" | "15m" | "30m" | "1h";

export type SymbolCode = "BTCUSDT" | "SOLUSDT";

export type ModelType = "logistic" | "ann";

export type SignalType = "long" | "short" | null;

export interface DashboardQuery {
  symbol: SymbolCode;
  timeframe: Timeframe;
  threshold: number;
  model: ModelType;
}

export interface PriceWithSignalPoint {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  signal: SignalType;
  maFast?: number | null;
  maSlow?: number | null;
}

export interface ProbabilityPoint {
  timestamp: string;
  probability: number;
  signalActive: boolean;
}

export interface RollingAUCPoint {
  timestamp: string;
  auc: number;
}

export interface EquityPoint {
  timestamp: string;
  equity: number;
}

export interface DrawdownPoint {
  timestamp: string;
  drawdown: number;
}

export interface FeatureImportancePoint {
  feature: string;
  importance: number;
}

export interface TradeDistributionPoint {
  bucket: string;
  count: number;
}

export interface DashboardData {
  priceSeries: PriceWithSignalPoint[];
  probabilitySeries: ProbabilityPoint[];
  rollingAucSeries: RollingAUCPoint[];
  equityCurve: EquityPoint[];
  drawdownSeries: DrawdownPoint[];
  featureImportance: FeatureImportancePoint[];
  tradeDistribution: TradeDistributionPoint[];
}
