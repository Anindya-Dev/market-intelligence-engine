"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import ChartWrapper from "@/components/ui/ChartWrapper";
import { FeatureImportancePoint } from "@/types/dashboard";

interface FeatureImportanceProps {
  data: FeatureImportancePoint[];
  loading: boolean;
  error: string | null;
}

export default function FeatureImportance({ data, loading, error }: FeatureImportanceProps) {
  return (
    <ChartWrapper loading={loading} error={error} height={220}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 8, right: 12, left: 22, bottom: 0 }}>
          <CartesianGrid stroke="#e5e7eb" />
          <XAxis type="number" tick={{ fill: "#6b7280", fontSize: 11 }} />
          <YAxis dataKey="feature" type="category" tick={{ fill: "#6b7280", fontSize: 11 }} width={110} />
          <Tooltip contentStyle={{ background: "#ffffff", border: "1px solid #d1d5db" }} />
          <Bar dataKey="importance" fill="#60a5fa" />
        </BarChart>
      </ResponsiveContainer>
    </ChartWrapper>
  );
}
