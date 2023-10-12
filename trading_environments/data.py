import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pprint import pprint
from typing import Callable, Optional, Protocol

import numpy as np
import pandas as pd
from freqtrade.strategy import merge_informative_pair, timeframe_to_minutes
from lazyft.data_loader import load_pair_data
from loguru import logger
from pyparsing import col


class DataInterface(Protocol):
    def get_data():
        pass

    def preprocess():
        pass

    def get_dates(self) -> list[pd.Timestamp]:
        pass

    def _add_return(df: pd.DataFrame, x: str, y: str):
        df[y] = df[x].pct_change()
        return df


class CustomMergeData(DataInterface):
    def __init__(self, pair: str, timeframes: list[str], timerange: str):
        self.m = MultiTimeFramePairData(pair, timeframes, timerange)

    def preprocess(
        self,
        x_column,
        y_column,
        func: Callable[[pd.DataFrame, str, str], pd.DataFrame] = None,
    ):
        if not func:
            func = self._add_return
        self.m.do_func_on_data(func, x_column, y_column)

    def get_data(self, column: str):
        return self.m.create_combined_data(column)

    def get_dates(self) -> list[pd.Timestamp]:
        return self.m.data[self.m.highest_timeframe].dates


class FreqtradeMergedData(DataInterface):
    pass


@dataclass
class PairData:
    pair: str
    timeframe: str
    timerange: str
    startup_candles: int = 0
    auto_load = True
    _data: Optional[pd.DataFrame] = None

    def __post_init__(self):
        if self.auto_load:
            self._data = load_pair_data(
                self.pair,
                self.timeframe,
                timerange=self.timerange,
                startup_candles=self.startup_candles,
            )

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def dates(self):
        return self.data["date"].drop_duplicates().tolist()

    def get_data_from_day(self, day: str):
        return self.data[self.data["date"] == day]

    # def get_data_from_date_range(self, start_date: str, end_date: str):
    #     return self.data[
    #         (self.data["date"] >= start_date) & (self.data["date"] <= end_date)
    #     ]

    def get_n_candles_from_date_range(self, start_date: str, end_date: str, n: int):
        return self.data[
            (self.data["date"] >= start_date) & (self.data["date"] < end_date)
        ].tail(n)

    def rolling_generator(self, n: int):
        for i in range(n, len(self.data)):
            slice = self.data.iloc[i : i + n]
            if len(slice) == n:
                yield slice
            else:
                logger.info(
                    f"Last slice is {len(slice)}, not {n}. Yielding tail of n={n}"
                )
                yield self.data.tail(n)
                break


@dataclass
class MultiTimeFramePairData:
    pair: str
    timeframes: list[str]
    timerange: str
    auto_load = True
    _data: dict[str, PairData] = field(default_factory=dict)

    def __post_init__(self):
        if self.auto_load:
            self._data = {
                timeframe: PairData(
                    self.pair,
                    timeframe,
                    self.timerange,
                    # startup_candles=self.startup_candles,
                )
                for timeframe in self.timeframes
            }

    @property
    def highest_timeframe(self):
        return max(self.timeframes, key=timeframe_to_minutes)

    @property
    def data(self):
        return self._data

    def create_combined_data(self, column: str, window=10) -> pd.DataFrame:
        timeframes = self.timeframes.copy()
        timeframes.remove(self.highest_timeframe)
        data = OrderedDict()
        dates = []

        gen = self.data[self.highest_timeframe].rolling_generator(window)
        while True:
            try:
                slice = next(gen)
            except StopIteration:
                break
            first_date = slice.iloc[0]["date"]
            last_date = slice.iloc[-1]["date"]
            dates.append(last_date)
            data[last_date] = [slice[column].to_list()]
            # logger.info(f"Merging {first_date} to {last_date}")
            for tf in timeframes:
                tf_data = self.data[tf].get_n_candles_from_date_range(
                    first_date, last_date, window
                )
                # logger.info(
                #     f"Included dates for {last_date}: {tf_data['date'].unique()}"
                # )
                data[last_date].append(tf_data[column].to_list())
            data[last_date] = np.array(data[last_date], dtype=np.float64).T

        return data

    def do_func_on_data(
        self,
        func: Callable[[pd.DataFrame, str, str], pd.DataFrame],
        column: str,
        new_column,
    ) -> None:
        for timeframe in self.timeframes:
            self.data[timeframe].data = func(
                self.data[timeframe].data, column, new_column
            )

    # def get_data_from_day(self, day: str, column: str) -> pd.DataFrame:
    #     merged = None
    #     last_tf = None
    #     for tf, pair_data in self.data.items():
    #         if not merged:
    #             merged = pair_data.get_data_from_day(day)[["date", column]]
    #         else:
    #             merged = merge_informative_pair(
    #                 merged,
    #                 pair_data.get_data_from_day(day)[["date", column]],
    #             )


class FreqTradeMergePairData(MultiTimeFramePairData):
    @property
    def lowest_timeframe(self):
        return min(self.timeframes, key=timeframe_to_minutes)

    def create_combined_data(self, column: str, window=10) -> pd.DataFrame:
        timeframes = self.timeframes.copy()
        timeframes.remove(self.lowest_timeframe)
        data = OrderedDict()
        dates = []

        gen = self.data[self.lowest_timeframe].rolling_generator(window)


if __name__ == "__main__":
    m = CustomMergeData("BTC/USDT", ["1h", "4h", "8h"], "20180101-20201231")
    m.preprocess("close", "return")
    data = m.get_data("close")
    # print(f"{array.shape} {len(dates)}")
    # print(dates[0], array[:, 0].tolist())

    # for d, v in data.items():
    #     print(d, v)
    print("Length of data: ", len(data))
    print("Size of data (MB): ", sys.getsizeof(data) / 1024**2)
    # print(array.T[0, 1].shape)
    # pd.json_normalize
