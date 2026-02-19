"use client";

import { useMemo, useState, lazy, Suspense } from "react";
import Card from "@/components/ui/Card";
import SectionTitle from "@/components/ui/SectionTitle";
import TopNavBar from "@/components/ui/TopNavBar";
import PriceChart from "@/components/charts/PriceChart";
import ProbabilityChart from "@/components/charts/ProbabilityChart";
import RollingAUCChart from "@/components/charts/RollingAUCChart";
import { useDashboardData, useMarketWebSocket } from "@/lib/hooks";
import { DashboardQuery, ModelType, SymbolCode, Timeframe } from "@/types/dashboard";

const EquityCurve = lazy(() => import("@/components/charts/EquityCurve"));
const DrawdownChart = lazy(() => import("@/components/charts/DrawdownChart"));
const FeatureImportance = lazy(() => import("@/components/charts/FeatureImportance"));
const TradeDistribution = lazy(() => import("@/components/charts/TradeDistribution"));

export default function DashboardPage() {
  const [symbol, setSymbol] = useState<SymbolCode>("BTCUSDT");
  const [timeframe, setTimeframe] = useState<Timeframe>("5m");
  const [threshold, setThreshold] = useState<number>(0.55);
  const [model, setModel] = useState<ModelType>("logistic");
  const [showMA] = useState<boolean>(false);

  const query = useMemo<DashboardQuery>(
    () => ({ symbol, timeframe, threshold, model }),
    [symbol, timeframe, threshold, model],
  );

  const { data, loading, error, refresh } = useDashboardData(query);
  const { connected } = useMarketWebSocket(symbol, timeframe);

  return (
    <main className="mx-auto min-h-screen w-full max-w-[1600px] p-4 md:p-6">
      <TopNavBar
        symbol={symbol}
        timeframe={timeframe}
        threshold={threshold}
        model={model}
        onSymbolChange={setSymbol}
        onTimeframeChange={setTimeframe}
        onThresholdChange={setThreshold}
        onModelChange={setModel}
        onRefresh={refresh}
      />

      <div className="mb-4 flex items-center justify-between text-xs text-muted">
        <span>Mode: Quant Research Terminal</span>
        <span className={connected ? "text-bull" : "text-muted"}>WebSocket: {connected ? "Connected" : "Idle"}</span>
      </div>

      <SectionTitle title="Market View" />
      <div className="mb-6">
        <Card title="Price, Signals, Volume">
          <PriceChart data={data?.priceSeries ?? []} loading={loading} error={error} showMA={showMA} />
        </Card>
      </div>

      <SectionTitle title="Model Intelligence" />
      <div className="mb-6 grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Card title="Predicted Probability">
          <ProbabilityChart
            data={data?.probabilitySeries ?? []}
            threshold={threshold}
            loading={loading}
            error={error}
          />
        </Card>

        <Card title="Rolling AUC (200)">
          <RollingAUCChart data={data?.rollingAucSeries ?? []} loading={loading} error={error} />
        </Card>
      </div>

      <SectionTitle title="Quant Diagnostics" />
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        <Card title="Equity Curve">
          <Suspense fallback={<div className="h-[220px] text-xs text-muted">Loading...</div>}>
            <EquityCurve data={data?.equityCurve ?? []} loading={loading} error={error} />
          </Suspense>
        </Card>

        <Card title="Drawdown">
          <Suspense fallback={<div className="h-[220px] text-xs text-muted">Loading...</div>}>
            <DrawdownChart data={data?.drawdownSeries ?? []} loading={loading} error={error} />
          </Suspense>
        </Card>

        <Card title="Feature Importance">
          <Suspense fallback={<div className="h-[220px] text-xs text-muted">Loading...</div>}>
            <FeatureImportance data={data?.featureImportance ?? []} loading={loading} error={error} />
          </Suspense>
        </Card>

        <Card title="Trade Distribution">
          <Suspense fallback={<div className="h-[220px] text-xs text-muted">Loading...</div>}>
            <TradeDistribution data={data?.tradeDistribution ?? []} loading={loading} error={error} />
          </Suspense>
        </Card>
      </div>
    </main>
  );
}
