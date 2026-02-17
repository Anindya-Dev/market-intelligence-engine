import pandas as pd

from app.strategies.base_strategy import BaseStrategy


class RSIStrategy(BaseStrategy):
    def __init__(self, rsi_window: int = 14, lower: float = 30.0, upper: float = 70.0) -> None:
        if rsi_window <= 0:
            raise ValueError("rsi_window must be positive")
        if lower >= upper:
            raise ValueError("lower must be less than upper")

        self.rsi_window = rsi_window
        self.lower = lower
        self.upper = upper

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if "close" not in data.columns:
            raise ValueError("DataFrame must contain a 'close' column")

        delta = data["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=self.rsi_window, min_periods=self.rsi_window).mean()
        avg_loss = loss.rolling(window=self.rsi_window, min_periods=self.rsi_window).mean()

        rs = avg_gain / avg_loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))

        signals = pd.Series("hold", index=data.index, dtype="object")
        signals.loc[(rsi < self.lower).fillna(False)] = "buy"
        signals.loc[(rsi > self.upper).fillna(False)] = "sell"
        return signals