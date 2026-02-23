"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import ChartWrapper from "@/components/ui/ChartWrapper";
import { TradeDistributionPoint } from "@/types/dashboard";

interface TradeDistributionProps {
  data: TradeDistributionPoint[];
  loading: boolean;
  error: string | null;
}

export default function TradeDistribution({ data, loading, error }: TradeDistributionProps) {
  return (
    <ChartWrapper loading={loading} error={error} height={220}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="#e5e7eb" />
          <XAxis dataKey="bucket" tick={{ fill: "#6b7280", fontSize: 11 }} />
          <YAxis tick={{ fill: "#6b7280", fontSize: 11 }} />
          <Tooltip contentStyle={{ background: "#ffffff", border: "1px solid #d1d5db" }} />
          <Bar dataKey="count" fill="#6b7280" />
        </BarChart>
      </ResponsiveContainer>
    </ChartWrapper>
  );
}
