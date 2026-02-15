import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import redis
import websockets
from sqlalchemy.exc import SQLAlchemyError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.append(str(BACKEND_DIR))


def load_backend_env() -> None:
    env_path = BACKEND_DIR / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


load_backend_env()

from app.database import SessionLocal, engine
from app.models.ohlc import OHLC1m

try:
    from app.services.redis_client import redis_client
except Exception:
    redis_client = redis.Redis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        decode_responses=True,
    )


STREAM_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade"


def minute_bucket_from_event_time(event_time_ms: int) -> datetime:
    event_dt = datetime.fromtimestamp(event_time_ms / 1000, tz=timezone.utc)
    return event_dt.replace(second=0, microsecond=0)


def print_candle(candle: dict) -> None:
    print(
        "BTCUSDT | "
        f"{candle['minute'].isoformat()} | "
        f"O: {candle['open']} H: {candle['high']} "
        f"L: {candle['low']} C: {candle['close']} V: {candle['volume']}"
    )


def safe_set_redis(key: str, value: str) -> None:
    try:
        redis_client.set(key, value)
    except Exception as exc:
        print(f"Redis write failed for {key}: {exc}")


def persist_candle(candle: dict) -> None:
    db = SessionLocal()
    try:
        row = OHLC1m(
            symbol="BTCUSDT",
            timestamp=candle["minute"].replace(tzinfo=None),
            open=candle["open"],
            high=candle["high"],
            low=candle["low"],
            close=candle["close"],
            volume=candle["volume"],
        )
        db.add(row)
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        print(f"DB insert failed: {exc}")
    finally:
        db.close()


async def stream_trades() -> None:
    reconnect_delay = 2
    candle = None

    while True:
        try:
            async with websockets.connect(STREAM_URL, ping_interval=20, ping_timeout=20) as ws:
                print("Connected to Binance trade stream.")

                async for message in ws:
                    data = json.loads(message)
                    price = data.get("p")
                    quantity = data.get("q")
                    event_time = data.get("E")

                    if price is None or quantity is None or event_time is None:
                        continue

                    trade_price = float(price)
                    trade_qty = float(quantity)
                    minute = minute_bucket_from_event_time(event_time)

                    safe_set_redis("btc_live_price", str(trade_price))

                    if candle is None:
                        candle = {
                            "minute": minute,
                            "open": trade_price,
                            "high": trade_price,
                            "low": trade_price,
                            "close": trade_price,
                            "volume": trade_qty,
                        }
                        continue

                    if minute != candle["minute"]:
                        print_candle(candle)
                        safe_set_redis("btc_last_candle", json.dumps({
                            "minute": candle["minute"].isoformat(),
                            "open": candle["open"],
                            "high": candle["high"],
                            "low": candle["low"],
                            "close": candle["close"],
                            "volume": candle["volume"],
                        }))
                        asyncio.create_task(asyncio.to_thread(persist_candle, candle.copy()))
                        candle = {
                            "minute": minute,
                            "open": trade_price,
                            "high": trade_price,
                            "low": trade_price,
                            "close": trade_price,
                            "volume": trade_qty,
                        }
                        continue

                    candle["high"] = max(candle["high"], trade_price)
                    candle["low"] = min(candle["low"], trade_price)
                    candle["close"] = trade_price
                    candle["volume"] += trade_qty

                reconnect_delay = 2
        except (websockets.WebSocketException, OSError, json.JSONDecodeError) as exc:
            print(f"Stream disconnected: {exc}. Reconnecting in {reconnect_delay}s...")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 30)


if __name__ == "__main__":
    asyncio.run(stream_trades())
