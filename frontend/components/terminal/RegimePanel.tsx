"use client";

import { RegimeState } from "@/lib/terminal_analytics";

interface RegimePanelProps {
  state: RegimeState;
}

function pct(v: number): string {
  return `${(v * 100).toFixed(2)}%`;
}

export default function RegimePanel({ state }: RegimePanelProps) {
  const tone =
    state.regime === "Trending"
      ? "text-bull"
      : state.regime === "High Volatility"
      ? "text-bear"
      : "text-muted";

  return (
    <div className="grid grid-cols-2 gap-3 text-xs md:grid-cols-4">
      <div className="rounded border border-border bg-slate-50 p-3">
        <div className="text-muted">Regime</div>
        <div className={`mt-1 text-sm font-semibold ${tone}`}>{state.regime}</div>
      </div>
      <div className="rounded border border-border bg-slate-50 p-3">
        <div className="text-muted">Rolling Volatility</div>
        <div className="mt-1 text-sm font-semibold">{pct(state.rollingVolatility)}</div>
      </div>
      <div className="rounded border border-border bg-slate-50 p-3">
        <div className="text-muted">Trend Strength</div>
        <div className="mt-1 text-sm font-semibold">{pct(state.trendStrength)}</div>
      </div>
      <div className="rounded border border-border bg-slate-50 p-3">
        <div className="text-muted">Model Confidence</div>
        <div className="mt-1 text-sm font-semibold">{pct(state.confidence)}</div>
      </div>
    </div>
  );
}
