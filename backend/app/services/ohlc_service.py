import math
import statistics

import pandas as pd
from sqlalchemy.orm import Session

from app.models.ohlc import OHLC1m


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