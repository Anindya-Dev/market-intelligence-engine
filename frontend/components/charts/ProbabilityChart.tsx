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
  secondaryData?: ProbabilityPoint[];
  threshold: number;
  loading: boolean;
  error: string | null;
}

export default function ProbabilityChart({
  data,
  secondaryData,
  threshold,
  loading,
  error,
}: ProbabilityChartProps) {
  const active = data.filter((d) => d.signalActive || d.probability >= threshold);

  const secondaryMap = new Map((secondaryData ?? []).map((d) => [d.timestamp, d.probability]));
  const merged = data.map((d) => ({
    ...d,
    probabilitySecondary: secondaryMap.get(d.timestamp) ?? null,
  }));

  return (
    <ChartWrapper loading={loading} error={error}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={merged} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="#e5e7eb" />
          <XAxis dataKey="timestamp" tick={{ fill: "#6b7280", fontSize: 11 }} minTickGap={26} />
          <YAxis domain={[0, 1]} tick={{ fill: "#6b7280", fontSize: 11 }} />
          <Tooltip contentStyle={{ background: "#ffffff", border: "1px solid #d1d5db" }} />
          <ReferenceLine y={0.5} stroke="#4b5563" strokeDasharray="4 4" />
          <ReferenceLine y={threshold} stroke="#22c55e" strokeDasharray="6 6" />
          <Line type="monotone" dataKey="probability" stroke="#60a5fa" dot={false} strokeWidth={2} />
          {secondaryData && secondaryData.length > 0 ? (
            <Line
              type="monotone"
              dataKey="probabilitySecondary"
              stroke="#f59e0b"
              dot={false}
              strokeWidth={1.6}
            />
          ) : null}
          <Scatter data={active} fill="#22c55e" />
        </LineChart>
      </ResponsiveContainer>
    </ChartWrapper>
  );
}
