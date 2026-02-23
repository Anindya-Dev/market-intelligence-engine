"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { fetchBackendHealth, fetchDashboardData, fetchProbabilitySeries } from "@/lib/api";
import { DashboardData, DashboardQuery, PriceWithSignalPoint, ProbabilityPoint } from "@/types/dashboard";

export interface UseDashboardDataResult {
  data: DashboardData | null;
  loading: boolean;
  error: string | null;
  refresh: () => void;
}

export function useDashboardData(query: DashboardQuery): UseDashboardDataResult {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [refreshToken, setRefreshToken] = useState<number>(0);

  const stableQuery = useMemo(() => query, [query]);

  const refresh = useCallback(() => {
    setRefreshToken((v) => v + 1);
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      setError(null);

      try {
        const payload = await fetchDashboardData(stableQuery);
        if (!cancelled) {
          setData(payload);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load dashboard data");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void load();

    return () => {
      cancelled = true;
    };
  }, [stableQuery, refreshToken]);

  return { data, loading, error, refresh };
}

interface CandleUpdateMessage {
  type: "candle_update";
  symbol: string;
  timeframe: string;
  event_time_ms: number;
  server_time: string;
  candle: {
    timestamp: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
  };
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isValidCandleShape(candle: CandleUpdateMessage["candle"]): boolean {
  if (!isFiniteNumber(candle.open) || candle.open <= 0) return false;
  if (!isFiniteNumber(candle.high) || candle.high <= 0) return false;
  if (!isFiniteNumber(candle.low) || candle.low <= 0) return false;
  if (!isFiniteNumber(candle.close) || candle.close <= 0) return false;
  if (!isFiniteNumber(candle.volume) || candle.volume < 0) return false;
  if (candle.high < candle.low) return false;
  if (candle.high < Math.max(candle.open, candle.close)) return false;
  if (candle.low > Math.min(candle.open, candle.close)) return false;
  return true;
}

export function useMarketWebSocket(
  symbol: string,
  timeframe: string,
  initialCandles: PriceWithSignalPoint[],
): {
  connected: boolean;
  candles: PriceWithSignalPoint[];
  latencyMs: number | null;
  lastUpdateMs: number | null;
  isStale: boolean;
} {
  const [connected, setConnected] = useState(false);
  const [candles, setCandles] = useState<PriceWithSignalPoint[]>(initialCandles);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [lastUpdateMs, setLastUpdateMs] = useState<number | null>(null);
  const [isStale, setIsStale] = useState<boolean>(false);

  useEffect(() => {
    const sanitized = initialCandles.filter((c) =>
      isValidCandleShape({
        timestamp: String(c.timestamp),
        open: Number(c.open),
        high: Number(c.high),
        low: Number(c.low),
        close: Number(c.close),
        volume: Number(c.volume),
      }),
    );

    if (sanitized.length !== initialCandles.length) {
      console.warn("[ws] dropped malformed initial candles", {
        received: initialCandles.length,
        kept: sanitized.length,
      });
    }

    setCandles(sanitized);
  }, [initialCandles]);

  useEffect(() => {
    const apiBase = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";
    const wsBase = process.env.NEXT_PUBLIC_WS_URL ?? `${apiBase.replace(/^http/, "ws")}/ws/market`;

    let ws: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let reconnectDelayMs = 1000;
    let isActive = true;

    const connect = () => {
      if (!isActive) return;
      ws = new WebSocket(`${wsBase}?symbol=${symbol}&timeframe=${timeframe}`);

      ws.onopen = () => {
        setConnected(true);
        reconnectDelayMs = 1000;
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data) as CandleUpdateMessage;
          if (msg.type !== "candle_update") return;

          console.debug("[ws] incoming candle_update", msg.candle);

          if (!isValidCandleShape(msg.candle)) {
            console.warn("[ws] ignored candle_update: invalid candle shape", msg.candle);
            return;
          }

          const now = Date.now();
          setLastUpdateMs(now);
          setIsStale(false);
          if (typeof msg.event_time_ms === "number") {
            setLatencyMs(Math.max(0, now - msg.event_time_ms));
          }

          const incoming: PriceWithSignalPoint = {
            timestamp: msg.candle.timestamp,
            open: msg.candle.open,
            high: msg.candle.high,
            low: msg.candle.low,
            close: msg.candle.close,
            volume: msg.candle.volume,
            signal: null,
            maFast: null,
            maSlow: null,
          };

          setCandles((prev) => {
            if (prev.length === 0) return [incoming];

            const last = prev[prev.length - 1];
            if (last.timestamp === incoming.timestamp) {
              const updated = [...prev];
              updated[updated.length - 1] = {
                ...incoming,
                signal: last.signal ?? null,
                maFast: last.maFast ?? null,
                maSlow: last.maSlow ?? null,
              };
              return updated;
            }

            const next = [...prev, incoming];
            if (next.length > 500) {
              next.splice(0, next.length - 500);
            }
            return next;
          });
        } catch {
          // ignore malformed messages
        }
      };

      ws.onerror = () => {
        setConnected(false);
      };

      ws.onclose = () => {
        setConnected(false);
        if (!isActive) return;
        reconnectTimer = setTimeout(connect, reconnectDelayMs);
        reconnectDelayMs = Math.min(reconnectDelayMs * 2, 15000);
      };
    };

    connect();

    return () => {
      isActive = false;
      setConnected(false);
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (ws && ws.readyState <= WebSocket.OPEN) {
        ws.close();
      }
    };
  }, [symbol, timeframe]);

  useEffect(() => {
    const timer = setInterval(() => {
      if (!lastUpdateMs) {
        setIsStale(true);
        return;
      }
      setIsStale(Date.now() - lastUpdateMs > 10_000);
    }, 1000);

    return () => clearInterval(timer);
  }, [lastUpdateMs]);

  return { connected, candles, latencyMs, lastUpdateMs, isStale };
}

export function useProbabilitySeries(query: DashboardQuery, enabled: boolean): ProbabilityPoint[] {
  const [series, setSeries] = useState<ProbabilityPoint[]>([]);

  useEffect(() => {
    let cancelled = false;
    if (!enabled) {
      setSeries([]);
      return;
    }

    async function load() {
      try {
        const data = await fetchProbabilitySeries(query);
        if (!cancelled) setSeries(data);
      } catch {
        if (!cancelled) setSeries([]);
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, [query, enabled]);

  return series;
}

export function useBackendHealthPing(intervalMs = 5000): { healthy: boolean; lastCheckedAt: number | null } {
  const [healthy, setHealthy] = useState(false);
  const [lastCheckedAt, setLastCheckedAt] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function check() {
      try {
        const response = await fetchBackendHealth();
        if (!cancelled) {
          setHealthy(response.status === "ok");
          setLastCheckedAt(Date.now());
        }
      } catch {
        if (!cancelled) {
          setHealthy(false);
          setLastCheckedAt(Date.now());
        }
      }
    }

    void check();
    const timer = setInterval(() => void check(), intervalMs);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [intervalMs]);

  return { healthy, lastCheckedAt };
}
