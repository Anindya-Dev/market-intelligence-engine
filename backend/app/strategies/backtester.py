import math

import numpy as np
import pandas as pd

from app.strategies.moving_average import MovingAverageStrategy


class Backtester:
    def __init__(self, data: pd.DataFrame, transaction_cost: float = 0.001) -> None:
        if "close" not in data.columns:
            raise ValueError("DataFrame must contain a 'close' column")
        if transaction_cost < 0:
            raise ValueError("transaction_cost must be non-negative")

        self.data = data.sort_values("timestamp").reset_index(drop=True).copy()
        self.transaction_cost = transaction_cost

    def run(self, strategy: MovingAverageStrategy, transaction_cost: float | None = None) -> dict:
        effective_cost = self.transaction_cost if transaction_cost is None else transaction_cost

        if len(self.data) < 2:
            return {
                "total_return": 0.0,
                "cagr": None,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "num_trades": 0,
                "data_points": len(self.data),
            }

        df = self.data.copy()
        df["signal"] = strategy.generate_signals(df)

        in_position = False
        positions = []
        cost_events = []

        for signal in df["signal"]:
            positions.append(1 if in_position else 0)
            trade_cost = 0.0
            if signal == "buy" and not in_position:
                in_position = True
                trade_cost = effective_cost
            elif signal == "sell" and in_position:
                in_position = False
                trade_cost = effective_cost
            cost_events.append(trade_cost)

        df["position"] = positions
        df["trade_cost"] = cost_events
        df["market_return"] = df["close"].pct_change().fillna(0.0)
        df["strategy_return"] = (df["market_return"] * df["position"]) - df["trade_cost"]
        df["equity"] = (1.0 + df["strategy_return"]).cumprod()
        final_equity = float(df["equity"].iloc[-1])

        total_return = float(final_equity - 1.0)

        periods = len(df)
        minutes_in_year = 365 * 24 * 60
        years = periods / minutes_in_year
        if years < (30 / 365):
            cagr = None
        else:
            cagr = float((final_equity ** (1 / years)) - 1.0)

        ret_std = float(df["strategy_return"].std(ddof=1))
        mean_ret = float(df["strategy_return"].mean())
        if ret_std < 1e-8:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = float((mean_ret / ret_std) * math.sqrt(365 * 24 * 60))

        rolling_max = df["equity"].cummax()
        drawdown = (df["equity"] / rolling_max) - 1.0
        max_drawdown = float(abs(drawdown.min()))

        trades = self._simulate_trades(df, transaction_cost=effective_cost)
        if trades:
            win_rate = float(np.mean([1.0 if t > 0 else 0.0 for t in trades]))
        else:
            win_rate = 0.0

        return {
            "total_return": total_return,
            "cagr": cagr,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "num_trades": len(trades),
            "data_points": len(df),
        }

    @staticmethod
    def _simulate_trades(df: pd.DataFrame, transaction_cost: float = 0.001) -> list[float]:
        in_position = False
        entry_price = 0.0
        trades = []

        for _, row in df.iterrows():
            signal = row["signal"]
            close = float(row["close"])

            if signal == "buy" and not in_position:
                in_position = True
                entry_price = close
            elif signal == "sell" and in_position:
                raw_return = (close - entry_price) / entry_price
                trades.append(raw_return - (2 * transaction_cost))
                in_position = False

        if in_position:
            final_close = float(df["close"].iloc[-1])
            raw_return = (final_close - entry_price) / entry_price
            trades.append(raw_return - (2 * transaction_cost))

        return trades


def run_moving_average_backtest(
    data: pd.DataFrame,
    short_window: int = 10,
    long_window: int = 30,
    transaction_cost: float = 0.001,
) -> dict:
    strategy = MovingAverageStrategy(short_window=short_window, long_window=long_window)
    backtester = Backtester(data=data, transaction_cost=transaction_cost)
    return backtester.run(strategy=strategy, transaction_cost=transaction_cost)
