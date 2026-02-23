import { PropsWithChildren } from "react";

interface ChartWrapperProps extends PropsWithChildren {
  height?: number;
  loading?: boolean;
  error?: string | null;
}

export default function ChartWrapper({
  children,
  height = 280,
  loading = false,
  error = null,
}: ChartWrapperProps) {
  if (loading) {
    return (
      <div className="flex items-center justify-center rounded border border-border bg-slate-50" style={{ height }}>
        <span className="text-xs text-muted">Loading...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center rounded border border-border bg-slate-50" style={{ height }}>
        <span className="text-xs text-bear">{error}</span>
      </div>
    );
  }

  return <div style={{ height }}>{children}</div>;
}
