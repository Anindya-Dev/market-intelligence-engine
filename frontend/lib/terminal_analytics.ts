import { PriceWithSignalPoint, ProbabilityPoint, Timeframe } from "@/types/dashboard";

export interface RegimeState {
  rollingVolatility: number;
  trendStrength: number;
  regime: "Trending" | "Ranging" | "High Volatility";
  confidence: number;
}

export interface SimTrade {
  entryPrice: number;
  entryTimestamp: string;
  qty: number;
}

export interface SimulationState {
  enabled: boolean;
  equity: number;
  pnl: number;
  openTrade: SimTrade | null;
  totalTrades: number;
  wins: number;
  expectancy: number;
  sharpe200: number;
  winRate: number;
}

function barsPerYear(timeframe: Timeframe): number {
  if (timeframe === "5m") return 365 * 24 * 12;
  if (timeframe === "15m") return 365 * 24 * 4;
  if (timeframe === "30m") return 365 * 24 * 2;
  return 365 * 24;
}

function std(values: number[]): number {
  if (values.length === 0) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

export function computeRegimeState(candles: PriceWithSignalPoint[], probabilities: ProbabilityPoint[]): RegimeState {
  if (candles.length < 30) {
    return {
      rollingVolatility: 0,
      trendStrength: 0,
      regime: "Ranging",
      confidence: 0,
    };
  }

  const closes = candles.map((c) => Number(c.close));
  const returns = closes.slice(1).map((c, i) => Math.log(c / closes[i]));
  const tailReturns = returns.slice(-30);
  const rollingVol = std(tailReturns) * Math.sqrt(30);

  const trendWindow = closes.slice(-50);
  const first = trendWindow[0];
  const last = trendWindow[trendWindow.length - 1];
  const trendStrength = first !== 0 ? Math.abs((last - first) / first) : 0;

  const probs = probabilities.slice(-50).map((p) => Number(p.probability));
  const confidence = probs.length
    ? probs.reduce((s, p) => s + Math.abs(p - 0.5) * 2, 0) / probs.length
    : 0;

  let regime: RegimeState["regime"] = "Ranging";
  if (rollingVol > 0.04) regime = "High Volatility";
  else if (trendStrength > 0.02) regime = "Trending";

  return {
    rollingVolatility: rollingVol,
    trendStrength,
    regime,
    confidence,
  };
}

export function runLivePaperSimulation(
  candles: PriceWithSignalPoint[],
  probabilities: ProbabilityPoint[],
  threshold: number,
  timeframe: Timeframe,
  enabled: boolean,
): SimulationState {
  const initialEquity = 100_000;
  if (!enabled || candles.length < 2) {
    return {
      enabled,
      equity: initialEquity,
      pnl: 0,
      openTrade: null,
      totalTrades: 0,
      wins: 0,
      expectancy: 0,
      sharpe200: 0,
      winRate: 0,
    };
  }

  const probMap = new Map(probabilities.map((p) => [String(p.timestamp), Number(p.probability)]));

  let equity = initialEquity;
  let openTrade: SimTrade | null = null;
  let totalTrades = 0;
  let wins = 0;
  const tradeReturns: number[] = [];
  const equityReturns: number[] = [];

  for (let i = 1; i < candles.length; i++) {
    const prev = candles[i - 1];
    const cur = candles[i];
    const prob = probMap.get(String(prev.timestamp));

    if (openTrade === null && prob !== undefined && prob >= threshold) {
      const qty = equity / Number(cur.open);
      openTrade = {
        entryPrice: Number(cur.open),
        entryTimestamp: String(cur.timestamp),
        qty,
      };
      continue;
    }

    if (openTrade !== null) {
      const shouldExit = prob !== undefined && prob < threshold;
      if (shouldExit || i === candles.length - 1) {
        const exitPrice = Number(cur.close);
        const tradeReturn = (exitPrice - openTrade.entryPrice) / openTrade.entryPrice;
        equity *= 1 + tradeReturn;
        tradeReturns.push(tradeReturn);
        if (tradeReturn > 0) wins += 1;
        totalTrades += 1;
        openTrade = null;
      }
    }

    const prevClose = Number(prev.close);
    const curClose = Number(cur.close);
    if (prevClose > 0 && curClose > 0) {
      equityReturns.push(Math.log(curClose / prevClose));
    }
  }

  const expectancy = tradeReturns.length ? tradeReturns.reduce((a, b) => a + b, 0) / tradeReturns.length : 0;
  const winRate = totalTrades ? wins / totalTrades : 0;

  const tail = equityReturns.slice(-200);
  const mu = tail.length ? tail.reduce((a, b) => a + b, 0) / tail.length : 0;
  const sigma = std(tail);
  const sharpe200 = sigma > 1e-12 ? (mu / sigma) * Math.sqrt(barsPerYear(timeframe)) : 0;

  return {
    enabled,
    equity,
    pnl: equity - initialEquity,
    openTrade,
    totalTrades,
    wins,
    expectancy,
    sharpe200,
    winRate,
  };
}
