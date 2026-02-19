"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { PriceWithSignalPoint } from "@/types/dashboard";
import ChartWrapper from "@/components/ui/ChartWrapper";

interface PriceChartProps {
  data: PriceWithSignalPoint[];
  loading: boolean;
  error: string | null;
  showMA: boolean;
}

function candlePath(
  x: number,
  open: number,
  close: number,
  high: number,
  low: number,
  candleWidth = 6,
): string {
  const bodyTop = Math.min(open, close);
  const bodyBottom = Math.max(open, close);
  return `M${x},${high} L${x},${bodyTop} M${x - candleWidth / 2},${bodyTop} L${x + candleWidth / 2},${bodyTop} L${x + candleWidth / 2},${bodyBottom} L${x - candleWidth / 2},${bodyBottom} Z M${x},${bodyBottom} L${x},${low}`;
}

function CandleShape(props: any) {
  const { cx, payload, yAxis } = props;

  if (typeof cx !== "number" || !payload || !yAxis?.scale) {
    return null;
  }

  const yScale = yAxis.scale;
  const openY = yScale(payload.open);
  const closeY = yScale(payload.close);
  const highY = yScale(payload.high);
  const lowY = yScale(payload.low);
  const up = payload.close >= payload.open;

  return (
    <path
      d={candlePath(cx, openY, closeY, highY, lowY)}
      fill={up ? "#22c55e" : "#ef4444"}
      stroke={up ? "#22c55e" : "#ef4444"}
      strokeWidth={1}
    />
  );
}

export default function PriceChart({ data, loading, error, showMA }: PriceChartProps) {
  const longSignals = data.filter((d) => d.signal === "long");
  const shortSignals = data.filter((d) => d.signal === "short");

  return (
    <div className="space-y-3">
      <ChartWrapper loading={loading} error={error} height={330}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 12, right: 18, bottom: 0, left: 6 }}>
            <CartesianGrid stroke="#1f2937" strokeDasharray="0" />
            <XAxis dataKey="timestamp" hide />
            <YAxis yAxisId="price" domain={["dataMin", "dataMax"]} tick={{ fill: "#9ca3af", fontSize: 11 }} />
            <Tooltip
              contentStyle={{ background: "#111827", border: "1px solid #1f2937" }}
              labelStyle={{ color: "#e5e7eb" }}
            />
            <Scatter yAxisId="price" data={data} dataKey="close" shape={<CandleShape />} />
            <Scatter yAxisId="price" data={longSignals} fill="#22c55e" shape="circle" />
            <Scatter yAxisId="price" data={shortSignals} fill="#ef4444" shape="circle" />
            {showMA ? null : null}
          </ComposedChart>
        </ResponsiveContainer>
      </ChartWrapper>

      <ChartWrapper loading={loading} error={error} height={160}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data}>
            <CartesianGrid stroke="#1f2937" />
            <XAxis dataKey="timestamp" tick={{ fill: "#9ca3af", fontSize: 11 }} minTickGap={26} />
            <YAxis tick={{ fill: "#9ca3af", fontSize: 11 }} />
            <Tooltip contentStyle={{ background: "#111827", border: "1px solid #1f2937" }} />
            <Bar dataKey="volume" fill="#374151" />
          </BarChart>
        </ResponsiveContainer>
      </ChartWrapper>
    </div>
  );
}
