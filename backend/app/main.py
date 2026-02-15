"""FastAPI application entrypoint (placeholder)."""

import json

from fastapi import Depends, FastAPI, Query
from sqlalchemy.orm import Session

from app.api.schemas import OHLCCreate, OHLCRead
from app.config import settings
from app.database import Base, engine, get_db
from app.models.ohlc import OHLC1m
from app.services.ohlc_service import (
    compute_ma_signal,
    compute_volatility,
    get_ohlc_data,
    get_ohlc_dataframe,
    resample_ohlc_dataframe,
)
from app.services.redis_client import get_value, set_value
from app.strategies.backtester import run_moving_average_backtest

app = FastAPI(title=settings.APP_NAME)


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/ohlc")
def create_ohlc(data: OHLCCreate, db: Session = Depends(get_db)):
    candle = OHLC1m(**data.model_dump())
    db.add(candle)
    db.commit()
    db.refresh(candle)
    return {"message": "OHLC inserted"}


@app.get("/ohlc", response_model=list[OHLCRead])
def read_ohlc(
    symbol: str | None = None,
    limit: int = Query(default=100, ge=1),
    db: Session = Depends(get_db),
):
    return get_ohlc_data(db=db, symbol=symbol, limit=limit)


@app.get("/redis-test")
def redis_test() -> dict:
    set_value("ping", "pong")
    value = get_value("ping")
    return {"ping": value}


@app.get("/live-price")
def live_price() -> dict:
    value = get_value("btc_live_price")
    if value is None:
        return {"symbol": "BTCUSDT", "price": None, "status": "missing"}

    try:
        return {"symbol": "BTCUSDT", "price": float(value), "status": "ok"}
    except ValueError:
        return {"symbol": "BTCUSDT", "price": value, "status": "invalid"}


@app.get("/last-candle")
def last_candle() -> dict:
    value = get_value("btc_last_candle")
    if value is None:
        return {"symbol": "BTCUSDT", "candle": None, "status": "missing"}

    try:
        candle = json.loads(value)
        return {"symbol": "BTCUSDT", "candle": candle, "status": "ok"}
    except json.JSONDecodeError:
        return {"symbol": "BTCUSDT", "candle": value, "status": "invalid"}


@app.get("/volatility")
def volatility(db: Session = Depends(get_db)) -> dict:
    window = 30
    value = compute_volatility(db=db, symbol="BTCUSDT", window=window)

    if value is None:
        return {"symbol": "BTCUSDT", "volatility": None, "window": window, "status": "insufficient_data"}

    return {"symbol": "BTCUSDT", "volatility": value, "window": window}


@app.get("/ma-signal")
def ma_signal(db: Session = Depends(get_db)) -> dict:
    result = compute_ma_signal(db=db, symbol="BTCUSDT", fetch_window=50, short_window=10, long_window=30)

    if result is None:
        return {
            "symbol": "BTCUSDT",
            "short_ma": None,
            "long_ma": None,
            "signal": "hold",
            "status": "insufficient_data",
        }

    return {
        "symbol": "BTCUSDT",
        "short_ma": result["short_ma"],
        "long_ma": result["long_ma"],
        "signal": result["signal"],
    }


@app.get("/backtest/ma")
def backtest_ma(
    short: int = Query(default=10, ge=1),
    long: int = Query(default=30, ge=2),
    cost: float = Query(default=0.001, ge=0.0),
    timeframe: str = Query(default="1m"),
    db: Session = Depends(get_db),
) -> dict:
    if short >= long:
        return {
            "symbol": "BTCUSDT",
            "short_window": short,
            "long_window": long,
            "transaction_cost": cost,
            "status": "invalid_parameters",
            "message": "short must be less than long",
        }

    data = get_ohlc_dataframe(db=db, symbol="BTCUSDT")

    if timeframe not in {"1m", "5m"}:
        return {
            "symbol": "BTCUSDT",
            "short_window": short,
            "long_window": long,
            "transaction_cost": cost,
            "timeframe": timeframe,
            "status": "invalid_parameters",
            "message": "timeframe must be '1m' or '5m'",
        }

    data = resample_ohlc_dataframe(data=data, timeframe=timeframe)

    if len(data) < 30:
        return {
            "symbol": "BTCUSDT",
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "short_window": short,
            "long_window": long,
            "transaction_cost": cost,
            "timeframe": timeframe,
            "status": "insufficient_data",
        }

    metrics = run_moving_average_backtest(
        data=data,
        short_window=short,
        long_window=long,
        transaction_cost=cost,
    )
    return {
        "symbol": "BTCUSDT",
        "short_window": short,
        "long_window": long,
        "transaction_cost": cost,
        "timeframe": timeframe,
        **metrics,
    }
