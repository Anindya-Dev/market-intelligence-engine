import math
import statistics

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.models.ohlc import OHLC1m
from app.strategies.backtester import Backtester, run_moving_average_backtest
from app.strategies.moving_average import MovingAverageStrategy
from app.strategies.rsi import RSIStrategy


def get_ohlc_data(db: Session, symbol: str | None = None, limit: int = 100) -> list[OHLC1m]:
    query = db.query(OHLC1m)

    if symbol:
        query = query.filter(OHLC1m.symbol == symbol)

    return query.order_by(OHLC1m.timestamp.desc()).limit(limit).all()


def compute_volatility(db: Session, symbol: str = "BTCUSDT", window: int = 30) -> float | None:
    candles = (
        db.query(OHLC1m)
        .filter(OHLC1m.symbol == symbol)
        .order_by(OHLC1m.timestamp.desc())
        .limit(window)
        .all()
    )

    if len(candles) < 2:
        return None

    closes = [row.close for row in reversed(candles)]
    returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes)) if closes[i - 1] > 0 and closes[i] > 0]

    if len(returns) < 2:
        return None

    vol = statistics.stdev(returns)
    return vol * math.sqrt(365 * 24 * 60)


def compute_ma_signal(
    db: Session,
    symbol: str = "BTCUSDT",
    fetch_window: int = 50,
    short_window: int = 10,
    long_window: int = 30,
) -> dict | None:
    candles = (
        db.query(OHLC1m)
        .filter(OHLC1m.symbol == symbol)
        .order_by(OHLC1m.timestamp.desc())
        .limit(fetch_window)
        .all()
    )

    if len(candles) < long_window:
        return None

    closes = [row.close for row in reversed(candles)]
    short_slice = closes[-short_window:]
    long_slice = closes[-long_window:]

    short_ma = sum(short_slice) / short_window
    long_ma = sum(long_slice) / long_window

    if short_ma > long_ma:
        signal = "buy"
    elif short_ma < long_ma:
        signal = "sell"
    else:
        signal = "hold"

    return {
        "short_ma": short_ma,
        "long_ma": long_ma,
        "signal": signal,
    }


def get_ohlc_dataframe(db: Session, symbol: str = "BTCUSDT", limit: int | None = None) -> pd.DataFrame:
    query = db.query(OHLC1m).filter(OHLC1m.symbol == symbol).order_by(OHLC1m.timestamp.asc())

    if limit is not None:
        query = query.limit(limit)

    candles = query.all()

    rows = [
        {
            "timestamp": row.timestamp,
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
            "volume": row.volume,
        }
        for row in candles
    ]

    return pd.DataFrame(rows)


def resample_ohlc_dataframe(data: pd.DataFrame, timeframe: str = "1m") -> pd.DataFrame:
    if timeframe == "1m":
        return data

    if timeframe != "5m":
        raise ValueError("Unsupported timeframe. Use '1m' or '5m'.")

    if data.empty:
        return data

    df = data.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    resampled = df.resample("5min").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    resampled = resampled.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return resampled


def backtest_ma_grid(
    db: Session,
    symbol: str = "BTCUSDT",
    timeframe: str = "5m",
    cost: float = 0.001,
    short_windows: list[int] | None = None,
    long_windows: list[int] | None = None,
    top_n: int = 10,
) -> dict:
    if short_windows is None:
        short_windows = [20, 50, 100]
    if long_windows is None:
        long_windows = [100, 150, 200, 300]

    data = get_ohlc_dataframe(db=db, symbol=symbol)
    data = resample_ohlc_dataframe(data=data, timeframe=timeframe)

    results: list[dict] = []

    for short in short_windows:
        for long in long_windows:
            if short >= long:
                continue

            metrics = run_moving_average_backtest(
                data=data,
                short_window=short,
                long_window=long,
                transaction_cost=cost,
            )

            results.append(
                {
                    "short_window": short,
                    "long_window": long,
                    "total_return": metrics["total_return"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "max_drawdown": metrics["max_drawdown"],
                    "win_rate": metrics["win_rate"],
                    "num_trades": metrics["num_trades"],
                }
            )

    sorted_results = sorted(results, key=lambda x: x["sharpe_ratio"], reverse=True)
    top_results = sorted_results[:top_n]

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "transaction_cost": cost,
        "tested_combinations": len(results),
        "results": top_results,
    }


def backtest_ma_walkforward(
    db: Session,
    symbol: str = "BTCUSDT",
    short_window: int = 10,
    long_window: int = 30,
    timeframe: str = "1m",
    cost: float = 0.001,
) -> dict:
    data = get_ohlc_dataframe(db=db, symbol=symbol)
    data = resample_ohlc_dataframe(data=data, timeframe=timeframe)

    if len(data) < long_window:
        return {
            "symbol": symbol,
            "short_window": short_window,
            "long_window": long_window,
            "timeframe": timeframe,
            "transaction_cost": cost,
            "train_sharpe": 0.0,
            "train_return": 0.0,
            "test_sharpe": 0.0,
            "test_return": 0.0,
            "test_drawdown": 0.0,
            "test_trades": 0,
            "status": "insufficient_data",
        }

    split_idx = int(len(data) * 0.7)
    train_df = data.iloc[:split_idx].reset_index(drop=True)
    test_df = data.iloc[split_idx:].reset_index(drop=True)

    train_metrics = run_moving_average_backtest(
        data=train_df,
        short_window=short_window,
        long_window=long_window,
        transaction_cost=cost,
    )
    test_metrics = run_moving_average_backtest(
        data=test_df,
        short_window=short_window,
        long_window=long_window,
        transaction_cost=cost,
    )

    return {
        "symbol": symbol,
        "short_window": short_window,
        "long_window": long_window,
        "timeframe": timeframe,
        "transaction_cost": cost,
        "train_sharpe": train_metrics["sharpe_ratio"],
        "train_return": train_metrics["total_return"],
        "test_sharpe": test_metrics["sharpe_ratio"],
        "test_return": test_metrics["total_return"],
        "test_drawdown": test_metrics["max_drawdown"],
        "test_trades": test_metrics["num_trades"],
    }


class _FixedSignalStrategy:
    def __init__(self, signals: pd.Series) -> None:
        self._signals = signals

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return self._signals.reindex(data.index).fillna("hold")


def backtest_ma_volfilter(
    db: Session,
    symbol: str = "BTCUSDT",
    short_window: int = 10,
    long_window: int = 30,
    timeframe: str = "1m",
    cost: float = 0.001,
    vol_window: int = 30,
) -> dict:
    data = get_ohlc_dataframe(db=db, symbol=symbol)
    data = resample_ohlc_dataframe(data=data, timeframe=timeframe)

    min_points = max(long_window, vol_window) + 1
    if len(data) < min_points:
        return {
            "symbol": symbol,
            "short_window": short_window,
            "long_window": long_window,
            "timeframe": timeframe,
            "transaction_cost": cost,
            "vol_window": vol_window,
            "vol_median": None,
            "total_return": 0.0,
            "cagr": None,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "num_trades": 0,
            "data_points": len(data),
            "status": "insufficient_data",
        }

    df = data.copy().reset_index(drop=True)
    log_return = np.log(df["close"] / df["close"].shift(1))
    rolling_vol = log_return.rolling(window=vol_window).std()
    vol_median = float(rolling_vol.median(skipna=True))
    vol_filter = rolling_vol > vol_median

    base_strategy = MovingAverageStrategy(short_window=short_window, long_window=long_window)
    raw_signals = base_strategy.generate_signals(df)
    filtered_signals = raw_signals.where(vol_filter.fillna(False), "hold")

    strategy = _FixedSignalStrategy(filtered_signals)
    backtester = Backtester(data=df, transaction_cost=cost)
    metrics = backtester.run(strategy=strategy, transaction_cost=cost)

    return {
        "symbol": symbol,
        "short_window": short_window,
        "long_window": long_window,
        "timeframe": timeframe,
        "transaction_cost": cost,
        "vol_window": vol_window,
        "vol_median": vol_median,
        **metrics,
    }


def backtest_rsi(
    db: Session,
    symbol: str = "BTCUSDT",
    rsi_window: int = 14,
    lower: float = 30.0,
    upper: float = 70.0,
    timeframe: str = "1m",
    cost: float = 0.001,
) -> dict:
    data = get_ohlc_dataframe(db=db, symbol=symbol)
    data = resample_ohlc_dataframe(data=data, timeframe=timeframe)

    if len(data) < rsi_window + 1:
        return {
            "symbol": symbol,
            "rsi_window": rsi_window,
            "lower": lower,
            "upper": upper,
            "timeframe": timeframe,
            "transaction_cost": cost,
            "total_return": 0.0,
            "cagr": None,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "num_trades": 0,
            "data_points": len(data),
            "status": "insufficient_data",
        }

    strategy = RSIStrategy(rsi_window=rsi_window, lower=lower, upper=upper)
    backtester = Backtester(data=data, transaction_cost=cost)
    metrics = backtester.run(strategy=strategy, transaction_cost=cost)

    return {
        "symbol": symbol,
        "rsi_window": rsi_window,
        "lower": lower,
        "upper": upper,
        "timeframe": timeframe,
        "transaction_cost": cost,
        **metrics,
    }


def backtest_rsi_with_trend_filter(
    db: Session,
    symbol: str = "BTCUSDT",
    rsi_window: int = 14,
    lower: float = 30.0,
    upper: float = 70.0,
    timeframe: str = "1m",
    cost: float = 0.001,
    trend_ma_window: int = 100,
    slope_threshold: float = 0.0001,
) -> dict:
    data = get_ohlc_dataframe(db=db, symbol=symbol)
    data = resample_ohlc_dataframe(data=data, timeframe=timeframe)

    min_points = max(rsi_window + 1, trend_ma_window + 1)
    if len(data) < min_points:
        return {
            "symbol": symbol,
            "rsi_window": rsi_window,
            "lower": lower,
            "upper": upper,
            "timeframe": timeframe,
            "transaction_cost": cost,
            "trend_ma_window": trend_ma_window,
            "slope_threshold": slope_threshold,
            "total_return": 0.0,
            "cagr": None,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "num_trades": 0,
            "data_points": len(data),
            "status": "insufficient_data",
        }

    df = data.copy().reset_index(drop=True)
    trend_ma = df["close"].rolling(window=trend_ma_window, min_periods=trend_ma_window).mean()
    ma_slope = trend_ma.pct_change()
    ranging_regime = ma_slope.abs() < slope_threshold

    rsi_strategy = RSIStrategy(rsi_window=rsi_window, lower=lower, upper=upper)
    rsi_signals = rsi_strategy.generate_signals(df)
    filtered_signals = rsi_signals.where(ranging_regime.fillna(False), "hold")

    strategy = _FixedSignalStrategy(filtered_signals)
    backtester = Backtester(data=df, transaction_cost=cost)
    metrics = backtester.run(strategy=strategy, transaction_cost=cost)

    return {
        "symbol": symbol,
        "rsi_window": rsi_window,
        "lower": lower,
        "upper": upper,
        "timeframe": timeframe,
        "transaction_cost": cost,
        "trend_ma_window": trend_ma_window,
        "slope_threshold": slope_threshold,
        **metrics,
    }