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
          <CartesianGrid stroke="#e5e7eb" />
          <XAxis dataKey="timestamp" tick={{ fill: "#6b7280", fontSize: 11 }} minTickGap={26} />
          <YAxis domain={[0.4, 0.7]} tick={{ fill: "#6b7280", fontSize: 11 }} />
          <Tooltip contentStyle={{ background: "#ffffff", border: "1px solid #d1d5db" }} />
          <Line type="monotone" dataKey="auc" stroke="#a78bfa" dot={false} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </ChartWrapper>
  );
}
