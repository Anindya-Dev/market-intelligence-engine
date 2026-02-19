"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import ChartWrapper from "@/components/ui/ChartWrapper";
import { ProbabilityPoint } from "@/types/dashboard";

interface ProbabilityChartProps {
  data: ProbabilityPoint[];
  threshold: number;
  loading: boolean;
  error: string | null;
}

export default function ProbabilityChart({ data, threshold, loading, error }: ProbabilityChartProps) {
  const active = data.filter((d) => d.signalActive || d.probability >= threshold);

  return (
    <ChartWrapper loading={loading} error={error}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="#1f2937" />
          <XAxis dataKey="timestamp" tick={{ fill: "#9ca3af", fontSize: 11 }} minTickGap={26} />
          <YAxis domain={[0, 1]} tick={{ fill: "#9ca3af", fontSize: 11 }} />
          <Tooltip contentStyle={{ background: "#111827", border: "1px solid #1f2937" }} />
          <ReferenceLine y={0.5} stroke="#4b5563" strokeDasharray="4 4" />
          <ReferenceLine y={threshold} stroke="#22c55e" strokeDasharray="6 6" />
          <Line type="monotone" dataKey="probability" stroke="#60a5fa" dot={false} strokeWidth={2} />
          <Scatter data={active} fill="#22c55e" />
        </LineChart>
      </ResponsiveContainer>
    </ChartWrapper>
  );
}
