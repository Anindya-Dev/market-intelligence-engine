import math
import statistics

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
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

    timeframe_to_rule = {
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
    }

    if timeframe not in timeframe_to_rule:
        raise ValueError("Unsupported timeframe. Use '1m', '5m', '15m', '30m', or '1h'.")

    if data.empty:
        return data

    df = data.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    resampled = df.resample(timeframe_to_rule[timeframe]).agg(
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


def _build_feature_frame(data: pd.DataFrame, forward_bars: int) -> pd.DataFrame:
    df = data.copy().reset_index(drop=True)

    df["forward_return"] = (df["close"].shift(-forward_bars) / df["close"]) - 1

    short_ma = df["close"].rolling(window=50, min_periods=50).mean()
    long_ma = df["close"].rolling(window=150, min_periods=150).mean()
    df["ma_spread"] = short_ma - long_ma

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    df["rsi"] = 100 - (100 / (1 + rs))

    log_ret = np.log(df["close"] / df["close"].shift(1))
    df["rolling_volatility"] = log_ret.rolling(window=30, min_periods=30).std()
    df["momentum_10"] = df["close"].pct_change(periods=10)
    df["volume_change"] = df["volume"].pct_change()

    df["momentum_1"] = df["close"].pct_change(periods=1)
    df["momentum_3"] = df["close"].pct_change(periods=3)
    df["momentum_6"] = df["close"].pct_change(periods=6)

    prev_close = df["close"].shift(1)
    tr_components = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    df["atr"] = true_range.rolling(window=14, min_periods=14).mean()

    df["trend_strength"] = (df["ma_spread"].abs() / df["close"].replace(0, np.nan))
    df["trend_strength_rolling_median"] = df["trend_strength"].rolling(window=100, min_periods=100).median()
    df["strong_trend_flag"] = (df["trend_strength"] > df["trend_strength_rolling_median"]).astype(int)

    vol_mean = df["rolling_volatility"].rolling(window=100, min_periods=100).mean()
    vol_std = df["rolling_volatility"].rolling(window=100, min_periods=100).std().replace(0, np.nan)
    df["vol_zscore"] = (df["rolling_volatility"] - vol_mean) / vol_std

    df["rsi_vol_interaction"] = df["rsi"] * df["rolling_volatility"]
    df["mom_trend_interaction"] = df["momentum_10"] * df["trend_strength"]
    df["vol_change_5"] = df["rolling_volatility"] - df["rolling_volatility"].shift(5)

    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["candle_body_ratio"] = (df["close"] - df["open"]).abs() / candle_range

    volume_mean = df["volume"].rolling(window=100, min_periods=100).mean()
    volume_std = df["volume"].rolling(window=100, min_periods=100).std().replace(0, np.nan)
    df["volume_zscore"] = (df["volume"] - volume_mean) / volume_std

    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna().copy()


def feature_analysis(
    db: Session,
    symbol: str = "BTCUSDT",
    timeframe: str = "5m",
    forward_bars: int = 1,
) -> dict:
    data = get_ohlc_dataframe(db=db, symbol=symbol)
    data = resample_ohlc_dataframe(data=data, timeframe=timeframe)

    if data.empty:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "forward_bars": forward_bars,
            "num_samples": 0,
            "correlations": {
                "ma_spread": None,
                "rsi": None,
                "rolling_volatility": None,
                "momentum_10": None,
                "volume_change": None,
                "atr": None,
            },
            "status": "insufficient_data",
        }

    feature_cols = ["ma_spread", "rsi", "rolling_volatility", "momentum_10", "volume_change", "atr"]
    analysis_df = _build_feature_frame(data=data, forward_bars=forward_bars)

    correlations = {
        col: (float(analysis_df["forward_return"].corr(analysis_df[col])) if len(analysis_df) > 1 else None)
        for col in feature_cols
    }

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "forward_bars": forward_bars,
        "num_samples": int(len(analysis_df)),
        "correlations": correlations,
    }


def direction_analysis(
    db: Session,
    symbol: str = "BTCUSDT",
    timeframe: str = "5m",
    forward_bars: int = 1,
) -> dict:
    data = get_ohlc_dataframe(db=db, symbol=symbol)
    data = resample_ohlc_dataframe(data=data, timeframe=timeframe)

    feature_cols = ["ma_spread", "rsi", "rolling_volatility", "momentum_10", "volume_change", "atr"]
    analysis_df = _build_feature_frame(data=data, forward_bars=forward_bars)

    if len(analysis_df) < 10:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "forward_bars": forward_bars,
            "num_samples": int(len(analysis_df)),
            "class_balance": {"up": None, "down": None},
            "auc": None,
            "accuracy": None,
            "feature_correlations": {k: None for k in feature_cols},
            "model_coefficients": {k: {"coefficient": None, "relative_importance": None} for k in feature_cols},
            "status": "insufficient_data",
        }

    analysis_df["direction"] = (analysis_df["forward_return"] > 0).astype(int)

    class_up = float(analysis_df["direction"].mean())
    class_down = float(1.0 - class_up)

    feature_correlations = {
        col: float(analysis_df["direction"].corr(analysis_df[col])) for col in feature_cols
    }

    split_idx = int(len(analysis_df) * 0.7)
    train_df = analysis_df.iloc[:split_idx]
    test_df = analysis_df.iloc[split_idx:]

    if train_df["direction"].nunique() < 2 or test_df["direction"].nunique() < 2:
        auc = None
        accuracy = None
        model_coefficients = {k: {"coefficient": None, "relative_importance": None} for k in feature_cols}
    else:
        model = LogisticRegression(max_iter=1000)
        model.fit(train_df[feature_cols], train_df["direction"])

        prob = model.predict_proba(test_df[feature_cols])[:, 1]
        pred = (prob >= 0.5).astype(int)

        auc = float(roc_auc_score(test_df["direction"], prob))
        accuracy = float(accuracy_score(test_df["direction"], pred))

        coefs = model.coef_[0]
        abs_sum = float(np.abs(coefs).sum())
        model_coefficients = {}
        for idx, col in enumerate(feature_cols):
            coef_val = float(coefs[idx])
            rel_importance = float(abs(coef_val) / abs_sum) if abs_sum > 0 else 0.0
            model_coefficients[col] = {
                "coefficient": coef_val,
                "relative_importance": rel_importance,
            }

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "forward_bars": forward_bars,
        "num_samples": int(len(analysis_df)),
        "class_balance": {"up": class_up, "down": class_down},
        "auc": auc,
        "accuracy": accuracy,
        "feature_correlations": feature_correlations,
        "model_coefficients": model_coefficients,
    }


def direction_threshold_backtest(
    db: Session,
    symbol: str = "BTCUSDT",
    timeframe: str = "5m",
    forward_bars: int = 1,
    probability_threshold: float = 0.55,
) -> dict:
    data = get_ohlc_dataframe(db=db, symbol=symbol)
    data = resample_ohlc_dataframe(data=data, timeframe=timeframe)

    feature_cols = ["ma_spread", "rsi", "rolling_volatility", "momentum_10", "volume_change", "atr"]
    analysis_df = _build_feature_frame(data=data, forward_bars=forward_bars)

    if len(analysis_df) < 10:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "forward_bars": forward_bars,
            "probability_threshold": probability_threshold,
            "num_samples": int(len(analysis_df)),
            "directional_accuracy": None,
            "hit_rate": None,
            "status": "insufficient_data",
        }

    analysis_df["direction"] = (analysis_df["forward_return"] > 0).astype(int)

    split_idx = int(len(analysis_df) * 0.7)
    train_df = analysis_df.iloc[:split_idx]
    test_df = analysis_df.iloc[split_idx:].copy()

    if train_df["direction"].nunique() < 2 or test_df["direction"].nunique() < 2:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "forward_bars": forward_bars,
            "probability_threshold": probability_threshold,
            "num_samples": int(len(analysis_df)),
            "directional_accuracy": None,
            "hit_rate": None,
            "status": "insufficient_class_variation",
        }

    model = LogisticRegression(max_iter=1000)
    model.fit(train_df[feature_cols], train_df["direction"])

    prob_up = model.predict_proba(test_df[feature_cols])[:, 1]
    test_df["prob_up"] = prob_up

    test_df["signal"] = 0
    test_df.loc[test_df["prob_up"] > probability_threshold, "signal"] = 1
    test_df.loc[test_df["prob_up"] < (1.0 - probability_threshold), "signal"] = -1

    active_df = test_df[test_df["signal"] != 0].copy()
    total_test = int(len(test_df))
    active_signals = int(len(active_df))

    if active_signals == 0:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "forward_bars": forward_bars,
            "probability_threshold": probability_threshold,
            "num_samples": int(len(analysis_df)),
            "test_samples": total_test,
            "active_signals": 0,
            "directional_accuracy": 0.0,
            "hit_rate": 0.0,
            "avg_return_long": None,
            "avg_return_short": None,
            "avg_return_overall": float(test_df["forward_return"].mean()) if total_test > 0 else None,
            "status": "no_signals",
        }

    actual_signed = np.where(active_df["direction"] == 1, 1, -1)
    correct = (active_df["signal"].to_numpy() == actual_signed)
    correct_count = int(correct.sum())

    hit_rate = float(correct_count / active_signals)
    directional_accuracy = float(correct_count / total_test)

    long_df = test_df[test_df["signal"] == 1]
    short_df = test_df[test_df["signal"] == -1]

    avg_return_long = float(long_df["forward_return"].mean()) if len(long_df) > 0 else None
    avg_return_short = float(short_df["forward_return"].mean()) if len(short_df) > 0 else None
    avg_return_overall = float(test_df["forward_return"].mean()) if total_test > 0 else None

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "forward_bars": forward_bars,
        "probability_threshold": probability_threshold,
        "num_samples": int(len(analysis_df)),
        "test_samples": total_test,
        "active_signals": active_signals,
        "directional_accuracy": directional_accuracy,
        "hit_rate": hit_rate,
        "avg_return_long": avg_return_long,
        "avg_return_short": avg_return_short,
        "avg_return_overall": avg_return_overall,
    }


def direction_long_only_eval(
    db: Session,
    symbol: str = "BTCUSDT",
    timeframe: str = "5m",
    forward_bars: int = 3,
    probability_threshold: float = 0.55,
    transaction_cost: float = 0.001,
) -> dict:
    data = get_ohlc_dataframe(db=db, symbol=symbol)
    data = resample_ohlc_dataframe(data=data, timeframe=timeframe)

    feature_cols = ["ma_spread", "rsi", "rolling_volatility", "momentum_10", "volume_change", "atr"]
    analysis_df = _build_feature_frame(data=data, forward_bars=forward_bars)

    if len(analysis_df) < 10:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "forward_bars": forward_bars,
            "probability_threshold": probability_threshold,
            "transaction_cost": transaction_cost,
            "num_samples": int(len(analysis_df)),
            "num_trades": 0,
            "win_rate": 0.0,
            "avg_return_per_trade": 0.0,
            "total_return": 0.0,
            "expectancy": 0.0,
            "avg_holding_return_before_cost": 0.0,
            "status": "insufficient_data",
        }

    analysis_df["direction"] = (analysis_df["forward_return"] > 0).astype(int)

    split_idx = int(len(analysis_df) * 0.7)
    train_df = analysis_df.iloc[:split_idx]
    test_df = analysis_df.iloc[split_idx:].copy()

    if train_df["direction"].nunique() < 2:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "forward_bars": forward_bars,
            "probability_threshold": probability_threshold,
            "transaction_cost": transaction_cost,
            "num_samples": int(len(analysis_df)),
            "num_trades": 0,
            "win_rate": 0.0,
            "avg_return_per_trade": 0.0,
            "total_return": 0.0,
            "expectancy": 0.0,
            "avg_holding_return_before_cost": 0.0,
            "status": "insufficient_class_variation",
        }

    model = LogisticRegression(max_iter=1000)
    model.fit(train_df[feature_cols], train_df["direction"])

    prob_up = model.predict_proba(test_df[feature_cols])[:, 1]
    test_df["prob_up"] = prob_up
    test_df["long_signal"] = test_df["prob_up"] > probability_threshold

    trades_df = test_df[test_df["long_signal"]].copy()
    num_trades = int(len(trades_df))

    if num_trades == 0:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "forward_bars": forward_bars,
            "probability_threshold": probability_threshold,
            "transaction_cost": transaction_cost,
            "num_samples": int(len(analysis_df)),
            "num_trades": 0,
            "win_rate": 0.0,
            "avg_return_per_trade": 0.0,
            "total_return": 0.0,
            "expectancy": 0.0,
            "avg_holding_return_before_cost": 0.0,
            "status": "no_long_signals",
        }

    raw_returns = trades_df["forward_return"].to_numpy(dtype=float)
    net_returns = raw_returns - (2.0 * transaction_cost)

    wins = net_returns > 0
    win_rate = float(wins.mean())
    avg_return_per_trade = float(net_returns.mean())
    total_return = float(np.prod(1.0 + net_returns) - 1.0)
    avg_holding_return_before_cost = float(raw_returns.mean())

    if wins.any():
        avg_win = float(net_returns[wins].mean())
    else:
        avg_win = 0.0

    if (~wins).any():
        avg_loss = float(net_returns[~wins].mean())
    else:
        avg_loss = 0.0

    expectancy = float((win_rate * avg_win) + ((1.0 - win_rate) * avg_loss))

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "forward_bars": forward_bars,
        "probability_threshold": probability_threshold,
        "transaction_cost": transaction_cost,
        "num_samples": int(len(analysis_df)),
        "num_trades": num_trades,
        "win_rate": win_rate,
        "avg_return_per_trade": avg_return_per_trade,
        "total_return": total_return,
        "expectancy": expectancy,
        "avg_holding_return_before_cost": avg_holding_return_before_cost,
    }


def direction_ann_analysis(
    db: Session,
    symbol: str = "BTCUSDT",
    timeframe: str = "5m",
    forward_bars: int = 3,
) -> dict:
    data = get_ohlc_dataframe(db=db, symbol=symbol)
    data = resample_ohlc_dataframe(data=data, timeframe=timeframe)

    analysis_df = _build_feature_frame(data=data, forward_bars=forward_bars)

    if len(analysis_df) < 10:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "forward_bars": forward_bars,
            "num_samples": int(len(analysis_df)),
            "logistic": {"auc": None, "accuracy": None},
            "ann": {"auc": None, "accuracy": None},
            "status": "insufficient_data",
        }

    analysis_df["direction"] = (analysis_df["forward_return"] > 0).astype(int)

    feature_cols = ["ma_spread", "rsi", "rolling_volatility", "momentum_10", "volume_change", "atr"]

    split_idx = int(len(analysis_df) * 0.7)
    train_df = analysis_df.iloc[:split_idx]
    test_df = analysis_df.iloc[split_idx:]

    if train_df["direction"].nunique() < 2 or test_df["direction"].nunique() < 2:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "forward_bars": forward_bars,
            "num_samples": int(len(analysis_df)),
            "logistic": {"auc": None, "accuracy": None},
            "ann": {"auc": None, "accuracy": None},
            "status": "insufficient_class_variation",
        }

    x_train = train_df[feature_cols]
    y_train = train_df["direction"]
    x_test = test_df[feature_cols]
    y_test = test_df["direction"]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    logistic = LogisticRegression(max_iter=1000)
    logistic.fit(x_train_scaled, y_train)
    logistic_prob = logistic.predict_proba(x_test_scaled)[:, 1]
    logistic_pred = (logistic_prob >= 0.5).astype(int)

    ann = MLPClassifier(
        hidden_layer_sizes=(16,),
        activation="relu",
        max_iter=500,
        random_state=42,
    )
    ann.fit(x_train_scaled, y_train)
    ann_prob = ann.predict_proba(x_test_scaled)[:, 1]
    ann_pred = (ann_prob >= 0.5).astype(int)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "forward_bars": forward_bars,
        "num_samples": int(len(analysis_df)),
        "logistic": {
            "auc": float(roc_auc_score(y_test, logistic_prob)),
            "accuracy": float(accuracy_score(y_test, logistic_pred)),
        },
        "ann": {
            "auc": float(roc_auc_score(y_test, ann_prob)),
            "accuracy": float(accuracy_score(y_test, ann_pred)),
        },
    }