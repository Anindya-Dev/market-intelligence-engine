from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return a signal series with values: buy, sell, hold."""
