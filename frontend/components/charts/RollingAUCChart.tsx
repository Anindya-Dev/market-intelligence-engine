"use client";

import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import ChartWrapper from "@/components/ui/ChartWrapper";
import { RollingAUCPoint } from "@/types/dashboard";

interface RollingAUCChartProps {
  data: RollingAUCPoint[];
  loading: boolean;
  error: string | null;
}

export default function RollingAUCChart({ data, loading, error }: RollingAUCChartProps) {
  return (
    <ChartWrapper loading={loading} error={error}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="#1f2937" />
          <XAxis dataKey="timestamp" tick={{ fill: "#9ca3af", fontSize: 11 }} minTickGap={26} />
          <YAxis domain={[0.4, 0.7]} tick={{ fill: "#9ca3af", fontSize: 11 }} />
          <Tooltip contentStyle={{ background: "#111827", border: "1px solid #1f2937" }} />
          <Line type="monotone" dataKey="auc" stroke="#a78bfa" dot={false} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </ChartWrapper>
  );
}
