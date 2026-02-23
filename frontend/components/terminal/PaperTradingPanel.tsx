"use client";

import { SimulationState } from "@/lib/terminal_analytics";

interface PaperTradingPanelProps {
  enabled: boolean;
  onEnabledChange: (enabled: boolean) => void;
  simulation: SimulationState;
}

function pct(v: number): string {
  return `${(v * 100).toFixed(2)}%`;
}

export default function PaperTradingPanel({
  enabled,
  onEnabledChange,
  simulation,
}: PaperTradingPanelProps) {
  return (
    <div className="space-y-3 text-xs">
      <label className="inline-flex items-center gap-2">
        <input type="checkbox" checked={enabled} onChange={(e) => onEnabledChange(e.target.checked)} />
        <span>Enable Live Paper Trading</span>
      </label>

      <div className="grid grid-cols-2 gap-2">
        <div className="rounded border border-border bg-slate-50 p-2">
          <div className="text-muted">Equity</div>
          <div className="mt-1 font-medium">${simulation.equity.toFixed(2)}</div>
        </div>
        <div className="rounded border border-border bg-slate-50 p-2">
          <div className="text-muted">PnL</div>
          <div className={`mt-1 font-medium ${simulation.pnl >= 0 ? "text-bull" : "text-bear"}`}>
            ${simulation.pnl.toFixed(2)}
          </div>
        </div>
        <div className="rounded border border-border bg-slate-50 p-2">
          <div className="text-muted">Trades</div>
          <div className="mt-1 font-medium">{simulation.totalTrades}</div>
        </div>
        <div className="rounded border border-border bg-slate-50 p-2">
          <div className="text-muted">Win Rate</div>
          <div className="mt-1 font-medium">{pct(simulation.winRate)}</div>
        </div>
      </div>

      {simulation.openTrade ? (
        <div className="rounded border border-border bg-slate-50 p-2">
          <div className="text-muted">Open Position</div>
          <div className="mt-1">Entry: ${simulation.openTrade.entryPrice.toFixed(2)}</div>
          <div className="text-muted">{new Date(simulation.openTrade.entryTimestamp).toLocaleString()}</div>
        </div>
      ) : (
        <div className="rounded border border-border bg-slate-50 p-2 text-muted">No open position</div>
      )}
    </div>
  );
}
