import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

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

from app.database import SessionLocal
from app.models.ohlc import OHLC1m


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
TARGET_CANDLES = 20_000
BATCH_LIMIT = 1_000


def fetch_klines_batch(limit: int, end_time_ms: int | None = None) -> list:
    query_params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": limit}
    if end_time_ms is not None:
        query_params["endTime"] = end_time_ms

    params = urlencode(query_params)
    with urlopen(f"{BINANCE_KLINES_URL}?{params}") as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def open_time_to_dt(open_time_ms: int) -> datetime:
    return datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc).replace(tzinfo=None)


def fetch_klines_batched(target_candles: int = TARGET_CANDLES) -> list:
    all_klines = []
    end_time_ms = None

    while len(all_klines) < target_candles:
        remaining = target_candles - len(all_klines)
        batch_size = min(BATCH_LIMIT, remaining)
        batch = fetch_klines_batch(limit=batch_size, end_time_ms=end_time_ms)

        if not batch:
            break

        all_klines.extend(batch)

        oldest_open_time = int(batch[0][0])
        end_time_ms = oldest_open_time - 1
        time.sleep(0.05)

    # Keep only exact target size and sort oldest -> newest.
    trimmed = all_klines[:target_candles]
    trimmed.sort(key=lambda k: int(k[0]))
    return trimmed


def chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def seed_historical() -> int:
    klines = fetch_klines_batched(target_candles=TARGET_CANDLES)
    db = SessionLocal()
    inserted = 0

    try:
        timestamps = [open_time_to_dt(int(k[0])) for k in klines]
        existing_timestamps = set()

        for ts_chunk in chunked(timestamps, 5_000):
            rows = (
                db.query(OHLC1m.timestamp)
                .filter(OHLC1m.symbol == SYMBOL, OHLC1m.timestamp.in_(ts_chunk))
                .all()
            )
            existing_timestamps.update(row[0] for row in rows)

        for kline in klines:
            ts = open_time_to_dt(int(kline[0]))
            if ts in existing_timestamps:
                continue

            row = OHLC1m(
                symbol=SYMBOL,
                timestamp=ts,
                open=float(kline[1]),
                high=float(kline[2]),
                low=float(kline[3]),
                close=float(kline[4]),
                volume=float(kline[5]),
            )
            db.add(row)
            inserted += 1

        db.commit()
        return inserted
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    inserted_rows = seed_historical()
    print(f"Inserted rows: {inserted_rows}")
