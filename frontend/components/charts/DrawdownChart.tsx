"use client";

import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import ChartWrapper from "@/components/ui/ChartWrapper";
import { DrawdownPoint } from "@/types/dashboard";

interface DrawdownChartProps {
  data: DrawdownPoint[];
  loading: boolean;
  error: string | null;
}

export default function DrawdownChart({ data, loading, error }: DrawdownChartProps) {
  return (
    <ChartWrapper loading={loading} error={error} height={220}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="#e5e7eb" />
          <XAxis dataKey="timestamp" hide />
          <YAxis tick={{ fill: "#6b7280", fontSize: 11 }} />
          <Tooltip contentStyle={{ background: "#ffffff", border: "1px solid #d1d5db" }} />
          <Area type="monotone" dataKey="drawdown" stroke="#ef4444" fill="#ef4444" fillOpacity={0.2} />
        </AreaChart>
      </ResponsiveContainer>
    </ChartWrapper>
  );
}
