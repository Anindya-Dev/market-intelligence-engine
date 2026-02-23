import asyncio
import copy
import json
import logging
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone

import websockets
from fastapi import WebSocket

logger = logging.getLogger(__name__)

TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
}


@dataclass
class CandleState:
    bucket_start: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketStreamHub:
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol.upper()
        self.timeframe = timeframe
        self.clients: set[WebSocket] = set()
        self._stream_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._lock = asyncio.Lock()
        self._recent_payloads: deque[dict] = deque(maxlen=5)

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.clients.add(websocket)
            if self._stream_task is None or self._stream_task.done():
                self._stop_event = asyncio.Event()
                self._stream_task = asyncio.create_task(self._run_stream())

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self.clients.discard(websocket)
            if not self.clients and self._stream_task and not self._stream_task.done():
                self._stop_event.set()
                self._stream_task.cancel()

    async def broadcast(self, payload: dict) -> None:
        if not self._is_valid_candle_payload(payload):
            logger.warning(
                "Skipping invalid candle payload for %s %s: %s",
                self.symbol,
                self.timeframe,
                payload,
            )
            return

        self._recent_payloads.append(copy.deepcopy(payload))
        logger.info(
            "Last %d candle_update payload(s) for %s %s: %s",
            len(self._recent_payloads),
            self.symbol,
            self.timeframe,
            json.dumps(list(self._recent_payloads), default=str),
        )

        disconnected: list[WebSocket] = []
        for client in list(self.clients):
            try:
                await client.send_json(payload)
            except Exception:
                disconnected.append(client)

        if disconnected:
            async with self._lock:
                for client in disconnected:
                    self.clients.discard(client)

    async def _run_stream(self) -> None:
        stream_url = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@trade"
        reconnect_delay = 1
        candle: CandleState | None = None
        last_emit_at = 0.0
        emit_interval_seconds = 1.0

        while not self._stop_event.is_set():
            try:
                async with websockets.connect(stream_url, ping_interval=20, ping_timeout=20) as ws:
                    reconnect_delay = 1
                    async for message in ws:
                        if self._stop_event.is_set():
                            break

                        try:
                            trade = json.loads(message)
                            price = float(trade["p"])
                            quantity = float(trade["q"])
                            event_time_ms = int(trade["E"])
                        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                            continue

                        bucket_start = self._bucket_start(event_time_ms)
                        if candle is None or bucket_start != candle.bucket_start:
                            if candle is not None and bucket_start > candle.bucket_start:
                                missing = self._build_missing_candles(
                                    previous_candle=candle,
                                    next_bucket_start=bucket_start,
                                )
                                for gap_candle in missing:
                                    await self.broadcast(
                                        self._build_candle_update(
                                            gap_candle,
                                            event_time_ms=gap_candle.bucket_start * 1000,
                                        )
                                    )

                            candle = CandleState(
                                bucket_start=bucket_start,
                                open=price,
                                high=price,
                                low=price,
                                close=price,
                                volume=quantity,
                            )
                            await self.broadcast(self._build_candle_update(candle, event_time_ms=event_time_ms))
                            last_emit_at = asyncio.get_running_loop().time()
                        else:
                            candle.high = max(candle.high, price)
                            candle.low = min(candle.low, price)
                            candle.close = price
                            candle.volume += quantity

                        now = asyncio.get_running_loop().time()
                        if now - last_emit_at >= emit_interval_seconds:
                            await self.broadcast(self._build_candle_update(candle, event_time_ms=event_time_ms))
                            last_emit_at = now
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 30)

    def _build_candle_update(self, candle: CandleState, event_time_ms: int) -> dict:
        return {
            "type": "candle_update",
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "event_time_ms": event_time_ms,
            "server_time": datetime.now(timezone.utc).isoformat(),
            "candle": {
                "timestamp": datetime.fromtimestamp(
                    candle.bucket_start,
                    tz=timezone.utc,
                ).isoformat(),
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
            },
        }

    def _build_missing_candles(
        self,
        previous_candle: CandleState,
        next_bucket_start: int,
    ) -> list[CandleState]:
        bucket_seconds = TIMEFRAME_SECONDS[self.timeframe]
        missing: list[CandleState] = []

        current = previous_candle.bucket_start + bucket_seconds
        while current < next_bucket_start:
            carry = previous_candle.close
            missing.append(
                CandleState(
                    bucket_start=current,
                    open=carry,
                    high=carry,
                    low=carry,
                    close=carry,
                    volume=0.0,
                )
            )
            current += bucket_seconds

        return missing

    def _is_valid_candle_payload(self, payload: dict) -> bool:
        if payload.get("type") != "candle_update":
            return False

        candle = payload.get("candle")
        if not isinstance(candle, dict):
            return False

        required_numeric_fields = ("open", "high", "low", "close", "volume")
        for field in required_numeric_fields:
            value = candle.get(field)
            if isinstance(value, bool):
                return False
            if not isinstance(value, (int, float)):
                return False
            if not math.isfinite(float(value)):
                return False

        o = float(candle["open"])
        h = float(candle["high"])
        l = float(candle["low"])
        c = float(candle["close"])
        v = float(candle["volume"])

        if o <= 0 or h <= 0 or l <= 0 or c <= 0:
            return False
        if v < 0:
            return False
        if h < max(o, c) or l > min(o, c) or h < l:
            return False

        timestamp = candle.get("timestamp")
        if not isinstance(timestamp, str) or not timestamp:
            return False

        event_time_ms = payload.get("event_time_ms")
        if isinstance(event_time_ms, bool) or not isinstance(event_time_ms, int):
            return False

        server_time = payload.get("server_time")
        if not isinstance(server_time, str) or not server_time:
            return False

        return True

    def _bucket_start(self, event_time_ms: int) -> int:
        seconds = event_time_ms // 1000
        bucket_seconds = TIMEFRAME_SECONDS[self.timeframe]
        return (seconds // bucket_seconds) * bucket_seconds


_hubs: dict[tuple[str, str], MarketStreamHub] = {}


def get_market_hub(symbol: str, timeframe: str) -> MarketStreamHub:
    key = (symbol.upper(), timeframe)
    if key not in _hubs:
        _hubs[key] = MarketStreamHub(symbol=symbol, timeframe=timeframe)
    return _hubs[key]
