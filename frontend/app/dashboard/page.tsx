"use client";

import { useMemo, useState, lazy, Suspense } from "react";
import Card from "@/components/ui/Card";
import SectionTitle from "@/components/ui/SectionTitle";
import TopNavBar from "@/components/ui/TopNavBar";
import PriceChart from "@/components/charts/PriceChart";
import ProbabilityChart from "@/components/charts/ProbabilityChart";
import RollingAUCChart from "@/components/charts/RollingAUCChart";
import DataHealthPanel from "@/components/terminal/DataHealthPanel";
import PaperTradingPanel from "@/components/terminal/PaperTradingPanel";
import PerformanceSummaryCard from "@/components/terminal/PerformanceSummaryCard";
import RegimePanel from "@/components/terminal/RegimePanel";
import { computeRegimeState, runLivePaperSimulation } from "@/lib/terminal_analytics";
import { useBackendHealthPing, useDashboardData, useProbabilitySeries } from "@/lib/hooks";
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
  const [paperEnabled, setPaperEnabled] = useState<boolean>(false);
  const [showMA] = useState<boolean>(false);
  const [telemetry, setTelemetry] = useState<{
    connected: boolean;
    latencyMs: number | null;
    lastUpdateMs: number | null;
    isStale: boolean;
  }>({
    connected: false,
    latencyMs: null,
    lastUpdateMs: null,
    isStale: true,
  });

  const query = useMemo<DashboardQuery>(
    () => ({ symbol, timeframe, threshold, model }),
    [symbol, timeframe, threshold, model],
  );

  const comparisonModel: ModelType = model === "logistic" ? "ann" : "logistic";
  const comparisonQuery = useMemo<DashboardQuery>(
    () => ({ symbol, timeframe, threshold, model: comparisonModel }),
    [symbol, timeframe, threshold, comparisonModel],
  );

  const { data, loading, error, refresh } = useDashboardData(query);
  const comparisonProbabilities = useProbabilitySeries(comparisonQuery, true);
  const { healthy: backendHealthy } = useBackendHealthPing(5000);

  const regimeState = useMemo(
    () => computeRegimeState(data?.priceSeries ?? [], data?.probabilitySeries ?? []),
    [data?.priceSeries, data?.probabilitySeries],
  );

  const simulation = useMemo(
    () =>
      runLivePaperSimulation(
        data?.priceSeries ?? [],
        data?.probabilitySeries ?? [],
        threshold,
        timeframe,
        paperEnabled,
      ),
    [data?.priceSeries, data?.probabilitySeries, threshold, timeframe, paperEnabled],
  );

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
      </div>

      <SectionTitle title="Market View" />
      <div className="mb-6">
        <Card title="Price, Signals, Volume">
          <PriceChart
            symbol={symbol}
            timeframe={timeframe}
            data={data?.priceSeries ?? []}
            probabilities={data?.probabilitySeries ?? []}
            comparisonProbabilities={comparisonProbabilities}
            threshold={threshold}
            loading={loading}
            error={error}
            showMA={showMA}
            onTelemetryChange={setTelemetry}
          />
        </Card>
      </div>

      <div className="mb-6 grid grid-cols-1 gap-4 xl:grid-cols-3">
        <Card title="Regime Intelligence">
          <RegimePanel state={regimeState} />
        </Card>
        <Card title="Data Health">
          <DataHealthPanel
            backendHealthy={backendHealthy}
            connected={telemetry.connected}
            isStale={telemetry.isStale}
            latencyMs={telemetry.latencyMs}
            lastUpdateMs={telemetry.lastUpdateMs}
          />
        </Card>
        <Card title="Paper Trading">
          <PaperTradingPanel enabled={paperEnabled} onEnabledChange={setPaperEnabled} simulation={simulation} />
        </Card>
      </div>

      <div className="mb-6">
        <Card title="Performance Summary">
          <PerformanceSummaryCard
            sharpe200={simulation.sharpe200}
            expectancy={simulation.expectancy}
            winRate={simulation.winRate}
          />
        </Card>
      </div>

      <SectionTitle title="Model Intelligence" />
      <div className="mb-6 grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Card title="Predicted Probability">
          <ProbabilityChart
            data={data?.probabilitySeries ?? []}
            secondaryData={comparisonProbabilities}
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
