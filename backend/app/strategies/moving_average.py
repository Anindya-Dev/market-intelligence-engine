import pandas as pd

from app.strategies.base_strategy import BaseStrategy


class MovingAverageStrategy(BaseStrategy):
    def __init__(self, short_window: int = 10, long_window: int = 30) -> None:
        if short_window <= 0 or long_window <= 0:
            raise ValueError("Window sizes must be positive")
        if short_window >= long_window:
            raise ValueError("short_window must be smaller than long_window")

        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if "close" not in data.columns:
            raise ValueError("DataFrame must contain a 'close' column")

        df = data.copy()
        df["short_ma"] = df["close"].rolling(window=self.short_window, min_periods=self.short_window).mean()
        df["long_ma"] = df["close"].rolling(window=self.long_window, min_periods=self.long_window).mean()

        signal = pd.Series("hold", index=df.index, dtype="object")

        buy_cross = (df["short_ma"] > df["long_ma"]) & (df["short_ma"].shift(1) <= df["long_ma"].shift(1))
        sell_cross = (df["short_ma"] < df["long_ma"]) & (df["short_ma"].shift(1) >= df["long_ma"].shift(1))

        signal.loc[buy_cross.fillna(False)] = "buy"
        signal.loc[sell_cross.fillna(False)] = "sell"

        return signal
