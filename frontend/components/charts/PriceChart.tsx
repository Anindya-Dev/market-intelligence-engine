"use client";

import {
  Bar,
  BarChart,
  Brush,
  CartesianGrid,
  ComposedChart,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useEffect, useMemo, useState } from "react";
import { PriceWithSignalPoint, ProbabilityPoint } from "@/types/dashboard";
import ChartWrapper from "@/components/ui/ChartWrapper";
import { useMarketWebSocket } from "@/lib/hooks";

interface PriceChartProps {
  symbol: string;
  timeframe: string;
  data: PriceWithSignalPoint[];
  probabilities?: ProbabilityPoint[];
  threshold: number;
  comparisonProbabilities?: ProbabilityPoint[];
  loading: boolean;
  error: string | null;
  showMA: boolean;
  onTelemetryChange?: (payload: {
    connected: boolean;
    latencyMs: number | null;
    lastUpdateMs: number | null;
    isStale: boolean;
  }) => void;
}

type ChartMode = "candles" | "line" | "heikin";

function toHeikinAshi(rows: PriceWithSignalPoint[]): PriceWithSignalPoint[] {
  if (rows.length === 0) return [];
  const result: PriceWithSignalPoint[] = [];
  let prevHaOpen = (rows[0].open + rows[0].close) / 2;
  let prevHaClose = (rows[0].open + rows[0].high + rows[0].low + rows[0].close) / 4;

  rows.forEach((r, idx) => {
    const haClose = (r.open + r.high + r.low + r.close) / 4;
    const haOpen = idx === 0 ? prevHaOpen : (prevHaOpen + prevHaClose) / 2;
    const haHigh = Math.max(r.high, haOpen, haClose);
    const haLow = Math.min(r.low, haOpen, haClose);

    result.push({
      ...r,
      open: haOpen,
      high: haHigh,
      low: haLow,
      close: haClose,
    });

    prevHaOpen = haOpen;
    prevHaClose = haClose;
  });

  return result;
}

function computeMA(rows: Array<{ close: number }>, window: number): Array<number | null> {
  return rows.map((_, idx) => {
    if (idx + 1 < window) return null;
    const slice = rows.slice(idx + 1 - window, idx + 1);
    return slice.reduce((s, r) => s + r.close, 0) / window;
  });
}

function computeRSI(rows: Array<{ close: number }>, window = 14): Array<number | null> {
  if (rows.length === 0) return [];
  const out: Array<number | null> = new Array(rows.length).fill(null);
  for (let i = window; i < rows.length; i++) {
    let gain = 0;
    let loss = 0;
    for (let j = i - window + 1; j <= i; j++) {
      const diff = rows[j].close - rows[j - 1].close;
      if (diff >= 0) gain += diff;
      else loss -= diff;
    }
    const avgGain = gain / window;
    const avgLoss = loss / window;
    if (avgLoss === 0) {
      out[i] = 100;
      continue;
    }
    const rs = avgGain / avgLoss;
    out[i] = 100 - 100 / (1 + rs);
  }
  return out;
}

function CandleShape(props: any) {
  const { cx, payload, yAxis } = props;
  if (typeof cx !== "number" || !payload || !yAxis?.scale) return null;

  const y = yAxis.scale;
  const openY = y(payload.open);
  const closeY = y(payload.close);
  const highY = y(payload.high);
  const lowY = y(payload.low);
  const up = payload.close >= payload.open;
  const w = 6;
  const top = Math.min(openY, closeY);
  const bottom = Math.max(openY, closeY);

  return (
    <g>
      <line x1={cx} x2={cx} y1={highY} y2={lowY} stroke={up ? "#22c55e" : "#ef4444"} strokeWidth={1} />
      <rect
        x={cx - w / 2}
        y={top}
        width={w}
        height={Math.max(1, bottom - top)}
        fill={up ? "#22c55e" : "#ef4444"}
      />
    </g>
  );
}

function formatTime(value: string): string {
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

function formatPrice(value: number): string {
  if (!Number.isFinite(value)) return "";
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function isValidSeriesPoint(point: PriceWithSignalPoint): boolean {
  const open = Number(point.open);
  const high = Number(point.high);
  const low = Number(point.low);
  const close = Number(point.close);
  const volume = Number(point.volume);

  if (!Number.isFinite(open) || open <= 0) return false;
  if (!Number.isFinite(high) || high <= 0) return false;
  if (!Number.isFinite(low) || low <= 0) return false;
  if (!Number.isFinite(close) || close <= 0) return false;
  if (!Number.isFinite(volume) || volume < 0) return false;
  if (high < low) return false;
  if (high < Math.max(open, close)) return false;
  if (low > Math.min(open, close)) return false;
  return true;
}

export default function PriceChart({
  symbol,
  timeframe,
  data,
  probabilities,
  threshold,
  comparisonProbabilities,
  loading,
  error,
  showMA,
  onTelemetryChange,
}: PriceChartProps) {
  const { connected, candles, latencyMs, lastUpdateMs, isStale } = useMarketWebSocket(symbol, timeframe, data);

  useEffect(() => {
    onTelemetryChange?.({ connected, latencyMs, lastUpdateMs, isStale });
  }, [connected, latencyMs, lastUpdateMs, isStale, onTelemetryChange]);

  const [chartMode, setChartMode] = useState<ChartMode>("candles");
  const [showMA50, setShowMA50] = useState(true);
  const [showMA150, setShowMA150] = useState(false);
  const [showRSI, setShowRSI] = useState(false);
  const [showSignals, setShowSignals] = useState(true);
  const [showShortSignals, setShowShortSignals] = useState(false);
  const [windowSize, setWindowSize] = useState(180);
  const [windowEnd, setWindowEnd] = useState<number | null>(null);

  const normalized = useMemo(
    () => {
      const filtered = candles.filter(isValidSeriesPoint);
      if (filtered.length !== candles.length) {
        console.warn("[chart] dropped malformed points before render", {
          received: candles.length,
          kept: filtered.length,
        });
      }

      return filtered.map((d) => ({
        ...d,
        timestamp: String(d.timestamp),
        open: Number(d.open),
        high: Number(d.high),
        low: Number(d.low),
        close: Number(d.close),
        volume: Number(d.volume),
      }));
    },
    [candles],
  );

  const probabilityMap = useMemo(() => {
    const map = new Map<string, number>();
    (probabilities ?? []).forEach((p) => map.set(String(p.timestamp), Number(p.probability)));
    return map;
  }, [probabilities]);

  const comparisonMap = useMemo(() => {
    const map = new Map<string, number>();
    (comparisonProbabilities ?? []).forEach((p) => map.set(String(p.timestamp), Number(p.probability)));
    return map;
  }, [comparisonProbabilities]);

  const series = useMemo(() => {
    const source = chartMode === "heikin" ? toHeikinAshi(normalized) : normalized;
    const ma50 = computeMA(source, 50);
    const ma150 = computeMA(source, 150);
    const rsi = computeRSI(source, 14);

    return source.map((row, idx) => ({
      ...row,
      idx,
      ma50: ma50[idx],
      ma150: ma150[idx],
      rsi: rsi[idx],
      probability: probabilityMap.get(String(row.timestamp)) ?? null,
      probabilitySecondary: comparisonMap.get(String(row.timestamp)) ?? null,
    }));
  }, [chartMode, normalized, probabilityMap, comparisonMap]);

  useEffect(() => {
    if (series.length === 0) return;
    setWindowEnd(series.length - 1);
  }, [series.length]);

  const safeWindowSize = Math.max(30, Math.min(windowSize, Math.max(30, series.length)));
  const end = windowEnd ?? Math.max(0, series.length - 1);
  const start = Math.max(0, end - safeWindowSize + 1);
  const visible = series.slice(start, end + 1);

  const signalPoints = useMemo(
    () =>
      visible
        .filter((r) => r.probability !== null)
        .filter((r) => (r.probability as number) >= threshold || (showShortSignals && (r.probability as number) < 1 - threshold))
        .map((r) => ({
          ...r,
          direction: (r.probability as number) >= threshold ? "long" : "short",
        })),
    [visible, threshold, showShortSignals],
  );

  const tickFormatter = (value: number) => formatTime(String(series[Math.round(value)]?.timestamp ?? ""));

  const handleZoomIn = () => setWindowSize((s) => Math.max(40, Math.floor(s * 0.8)));
  const handleZoomOut = () => setWindowSize((s) => Math.min(Math.max(series.length, 60), Math.floor(s * 1.25)));
  const handlePanLeft = () => setWindowEnd((e) => (e === null ? null : Math.max(safeWindowSize - 1, e - Math.floor(safeWindowSize * 0.2))));
  const handlePanRight = () =>
    setWindowEnd((e) => (e === null ? null : Math.min(series.length - 1, e + Math.floor(safeWindowSize * 0.2))));

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <button className="rounded border border-border px-2 py-1" onClick={handleZoomIn} type="button">Zoom +</button>
        <button className="rounded border border-border px-2 py-1" onClick={handleZoomOut} type="button">Zoom -</button>
        <button className="rounded border border-border px-2 py-1" onClick={handlePanLeft} type="button">◀ Pan</button>
        <button className="rounded border border-border px-2 py-1" onClick={handlePanRight} type="button">Pan ▶</button>

        <select className="rounded border border-border px-2 py-1" value={chartMode} onChange={(e) => setChartMode(e.target.value as ChartMode)}>
          <option value="candles">Candles</option>
          <option value="line">Line (close)</option>
          <option value="heikin">Heikin Ashi</option>
        </select>

        <label className="inline-flex items-center gap-1"><input type="checkbox" checked={showMA50} onChange={(e) => setShowMA50(e.target.checked)} />MA(50)</label>
        <label className="inline-flex items-center gap-1"><input type="checkbox" checked={showMA150} onChange={(e) => setShowMA150(e.target.checked)} />MA(150)</label>
        <label className="inline-flex items-center gap-1"><input type="checkbox" checked={showRSI} onChange={(e) => setShowRSI(e.target.checked)} />RSI</label>
        <label className="inline-flex items-center gap-1"><input type="checkbox" checked={showSignals} onChange={(e) => setShowSignals(e.target.checked)} />Signals</label>
      </div>

      <div className="flex items-center justify-end text-xs">
        <span className={connected ? "text-bull" : "text-bear"}>{connected ? "Live" : "Disconnected"}</span>
        <span className="ml-3 text-muted">Latency: {latencyMs ?? "-"} ms</span>
        <span className={isStale ? "ml-3 text-bear" : "ml-3 text-muted"}>
          Last: {lastUpdateMs ? new Date(lastUpdateMs).toLocaleTimeString() : "-"}
        </span>
      </div>

      <ChartWrapper loading={loading} error={error} height={330}>
        <ResponsiveContainer width="100%" height="100%">
          {chartMode === "candles" || chartMode === "heikin" ? (
            <ComposedChart data={visible} margin={{ top: 12, right: 18, bottom: 0, left: 6 }}>
              <CartesianGrid stroke="#e5e7eb" strokeDasharray="0" />
              <XAxis dataKey="idx" type="number" domain={[start, end]} tickFormatter={tickFormatter} tick={{ fill: "#6b7280", fontSize: 11 }} />
              <YAxis domain={["dataMin", "dataMax"]} tick={{ fill: "#6b7280", fontSize: 11 }} tickFormatter={(v) => formatPrice(Number(v))} />
              <Tooltip
                cursor={{ stroke: "#9ca3af", strokeWidth: 1 }}
                contentStyle={{ background: "#ffffff", border: "1px solid #d1d5db" }}
                labelFormatter={(label) => formatTime(String(series[Math.round(Number(label))]?.timestamp ?? ""))}
              />
              <Scatter data={visible} dataKey="close" shape={<CandleShape />} isAnimationActive={false} />
              {showMA50 ? <Line type="linear" dataKey="ma50" stroke="#2563eb" dot={false} isAnimationActive={false} /> : null}
              {showMA150 ? <Line type="linear" dataKey="ma150" stroke="#7c3aed" dot={false} isAnimationActive={false} /> : null}
              {showSignals ? (
                <Scatter
                  data={signalPoints}
                  dataKey="close"
                  shape="triangle"
                  fill="#22c55e"
                  isAnimationActive={false}
                />
              ) : null}
              <Brush dataKey="idx" height={20} stroke="#94a3b8" startIndex={start} endIndex={end} />
            </ComposedChart>
          ) : (
            <LineChart data={visible} margin={{ top: 12, right: 18, bottom: 0, left: 6 }}>
              <CartesianGrid stroke="#e5e7eb" strokeDasharray="0" />
              <XAxis dataKey="idx" type="number" domain={[start, end]} tickFormatter={tickFormatter} tick={{ fill: "#6b7280", fontSize: 11 }} />
              <YAxis domain={["dataMin", "dataMax"]} tick={{ fill: "#6b7280", fontSize: 11 }} tickFormatter={(v) => formatPrice(Number(v))} />
              <Tooltip
                cursor={{ stroke: "#9ca3af", strokeWidth: 1 }}
                contentStyle={{ background: "#ffffff", border: "1px solid #d1d5db" }}
                labelFormatter={(label) => formatTime(String(series[Math.round(Number(label))]?.timestamp ?? ""))}
              />
              <Line type="linear" dataKey="close" stroke="#60a5fa" dot={false} isAnimationActive={false} />
              {showMA50 ? <Line type="linear" dataKey="ma50" stroke="#2563eb" dot={false} isAnimationActive={false} /> : null}
              {showMA150 ? <Line type="linear" dataKey="ma150" stroke="#7c3aed" dot={false} isAnimationActive={false} /> : null}
              {showSignals ? (
                <Scatter
                  data={signalPoints}
                  dataKey="close"
                  shape="triangle"
                  fill="#22c55e"
                  isAnimationActive={false}
                />
              ) : null}
              <Brush dataKey="idx" height={20} stroke="#94a3b8" startIndex={start} endIndex={end} />
            </LineChart>
          )}
        </ResponsiveContainer>
      </ChartWrapper>

      {showRSI ? (
        <ChartWrapper loading={loading} error={error} height={120}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={visible}>
              <CartesianGrid stroke="#e5e7eb" />
              <XAxis dataKey="idx" type="number" domain={[start, end]} tickFormatter={tickFormatter} tick={{ fill: "#6b7280", fontSize: 10 }} />
              <YAxis domain={[0, 100]} tick={{ fill: "#6b7280", fontSize: 10 }} />
              <Tooltip contentStyle={{ background: "#ffffff", border: "1px solid #d1d5db" }} />
              <ReferenceLine y={30} stroke="#ef4444" strokeDasharray="4 4" />
              <ReferenceLine y={70} stroke="#22c55e" strokeDasharray="4 4" />
              <Line type="linear" dataKey="rsi" stroke="#f59e0b" dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartWrapper>
      ) : null}

      <ChartWrapper loading={loading} error={error} height={160}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={visible}>
            <CartesianGrid stroke="#e5e7eb" />
            <XAxis
              dataKey="idx"
              type="number"
              domain={[start, end]}
              tick={{ fill: "#6b7280", fontSize: 11 }}
              minTickGap={26}
              tickFormatter={(v) => tickFormatter(Number(v))}
            />
            <YAxis tick={{ fill: "#6b7280", fontSize: 11 }} />
            <Tooltip
              contentStyle={{ background: "#ffffff", border: "1px solid #d1d5db" }}
              labelFormatter={(label) => tickFormatter(Number(label))}
            />
            <Bar dataKey="volume" fill="#94a3b8" />
          </BarChart>
        </ResponsiveContainer>
      </ChartWrapper>
    </div>
  );
}
