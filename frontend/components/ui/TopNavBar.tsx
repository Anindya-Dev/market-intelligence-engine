"use client";

import { ModelType, SymbolCode, Timeframe } from "@/types/dashboard";

interface TopNavBarProps {
  symbol: SymbolCode;
  timeframe: Timeframe;
  threshold: number;
  model: ModelType;
  onSymbolChange: (value: SymbolCode) => void;
  onTimeframeChange: (value: Timeframe) => void;
  onThresholdChange: (value: number) => void;
  onModelChange: (value: ModelType) => void;
  onRefresh: () => void;
}

export default function TopNavBar({
  symbol,
  timeframe,
  threshold,
  model,
  onSymbolChange,
  onTimeframeChange,
  onThresholdChange,
  onModelChange,
  onRefresh,
}: TopNavBarProps) {
  return (
    <div className="sticky top-0 z-20 mb-4 rounded-lg border border-border bg-panel px-4 py-3 shadow-sm">
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-6">
        <label className="flex flex-col gap-1 text-xs text-muted">
          Symbol
          <select
            className="rounded border border-border bg-white px-2 py-1 text-sm text-gray-900"
            value={symbol}
            onChange={(e) => onSymbolChange(e.target.value as SymbolCode)}
          >
            <option value="BTCUSDT">BTCUSDT</option>
            <option value="SOLUSDT">SOLUSDT</option>
          </select>
        </label>

        <label className="flex flex-col gap-1 text-xs text-muted">
          Timeframe
          <select
            className="rounded border border-border bg-white px-2 py-1 text-sm text-gray-900"
            value={timeframe}
            onChange={(e) => onTimeframeChange(e.target.value as Timeframe)}
          >
            <option value="5m">5m</option>
            <option value="15m">15m</option>
            <option value="30m">30m</option>
            <option value="1h">1h</option>
          </select>
        </label>

        <label className="flex flex-col gap-1 text-xs text-muted xl:col-span-2">
          Probability Threshold: {threshold.toFixed(2)}
          <input
            type="range"
            min={0.5}
            max={0.7}
            step={0.01}
            value={threshold}
            onChange={(e) => onThresholdChange(Number(e.target.value))}
            className="accent-bull"
          />
        </label>

        <label className="flex flex-col gap-1 text-xs text-muted">
          Model
          <select
            className="rounded border border-border bg-white px-2 py-1 text-sm text-gray-900"
            value={model}
            onChange={(e) => onModelChange(e.target.value as ModelType)}
          >
            <option value="logistic">Logistic</option>
            <option value="ann">ANN (placeholder)</option>
          </select>
        </label>

        <div className="flex items-end">
          <button
            type="button"
            onClick={onRefresh}
            className="w-full rounded border border-border bg-white px-3 py-2 text-sm text-gray-900 hover:bg-slate-50"
          >
            Refresh
          </button>
        </div>
      </div>
    </div>
  );
}
