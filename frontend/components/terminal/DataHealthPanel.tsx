"use client";

interface DataHealthPanelProps {
  backendHealthy: boolean;
  connected: boolean;
  isStale: boolean;
  latencyMs: number | null;
  lastUpdateMs: number | null;
}

function dotClass(ok: boolean): string {
  return ok ? "bg-bull" : "bg-bear";
}

export default function DataHealthPanel({
  backendHealthy,
  connected,
  isStale,
  latencyMs,
  lastUpdateMs,
}: DataHealthPanelProps) {
  return (
    <div className="space-y-2 text-xs">
      <div className="flex items-center justify-between rounded border border-border bg-slate-50 p-2">
        <span>Backend API</span>
        <span className="inline-flex items-center gap-2">
          <span className={`h-2 w-2 rounded-full ${dotClass(backendHealthy)}`} />
          <span>{backendHealthy ? "Healthy" : "Down"}</span>
        </span>
      </div>

      <div className="flex items-center justify-between rounded border border-border bg-slate-50 p-2">
        <span>Market Stream</span>
        <span className="inline-flex items-center gap-2">
          <span className={`h-2 w-2 rounded-full ${dotClass(connected && !isStale)}`} />
          <span>{connected ? (isStale ? "Stale" : "Live") : "Disconnected"}</span>
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div className="rounded border border-border bg-slate-50 p-2">
          <div className="text-muted">Latency</div>
          <div className="mt-1 font-medium">{latencyMs ?? "-"} ms</div>
        </div>
        <div className="rounded border border-border bg-slate-50 p-2">
          <div className="text-muted">Last Update</div>
          <div className="mt-1 font-medium">{lastUpdateMs ? new Date(lastUpdateMs).toLocaleTimeString() : "-"}</div>
        </div>
      </div>
    </div>
  );
}
