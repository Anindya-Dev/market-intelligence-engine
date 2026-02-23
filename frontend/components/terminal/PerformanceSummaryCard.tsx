"use client";

interface PerformanceSummaryCardProps {
  sharpe200: number;
  expectancy: number;
  winRate: number;
}

function pct(v: number): string {
  return `${(v * 100).toFixed(2)}%`;
}

export default function PerformanceSummaryCard({
  sharpe200,
  expectancy,
  winRate,
}: PerformanceSummaryCardProps) {
  return (
    <div className="grid grid-cols-3 gap-2 text-xs">
      <div className="rounded border border-border bg-slate-50 p-2">
        <div className="text-muted">Sharpe (200)</div>
        <div className="mt-1 text-sm font-semibold">{sharpe200.toFixed(2)}</div>
      </div>
      <div className="rounded border border-border bg-slate-50 p-2">
        <div className="text-muted">Expectancy</div>
        <div className={`mt-1 text-sm font-semibold ${expectancy >= 0 ? "text-bull" : "text-bear"}`}>{pct(expectancy)}</div>
      </div>
      <div className="rounded border border-border bg-slate-50 p-2">
        <div className="text-muted">Win Rate</div>
        <div className="mt-1 text-sm font-semibold">{pct(winRate)}</div>
      </div>
    </div>
  );
}
