"use client";

import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import ChartWrapper from "@/components/ui/ChartWrapper";
import { EquityPoint } from "@/types/dashboard";

interface EquityCurveProps {
  data: EquityPoint[];
  loading: boolean;
  error: string | null;
}

export default function EquityCurve({ data, loading, error }: EquityCurveProps) {
  return (
    <ChartWrapper loading={loading} error={error} height={220}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="#e5e7eb" />
          <XAxis dataKey="timestamp" hide />
          <YAxis tick={{ fill: "#6b7280", fontSize: 11 }} />
          <Tooltip contentStyle={{ background: "#ffffff", border: "1px solid #d1d5db" }} />
          <Line type="monotone" dataKey="equity" stroke="#22c55e" dot={false} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </ChartWrapper>
  );
}
