"""Based on https://github.com/hugocen/freqtrade-gym/blob/master/freqtradegym.py"""
import datetime
import sys
import time
from collections import OrderedDict, defaultdict
from enum import Enum
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Optional

import gym
import numpy as np
import pandas as pd
import wandb
from freqtrade.persistence import LocalTrade
from gym import spaces
from gym.envs.registration import register
from loguru import logger
from pandas import Timestamp
from torch.utils.tensorboard import SummaryWriter

from trading_environments.util import (
    calc_average_reward,
    calc_win_ratio,
    get_average_ratio,
    get_average_trade_duration,
    get_total_profit_percent,
)

# create an Dic


class Actions(Enum):
    Hold = 0
    Buy = 1
    Sell = 2


# class PercentOfBalance(Enum):
#     for i in range(1, 10):
#         locals()["PercentOfBalance" + str(i)] = i


def log_writer(log_queue):
    logger.info("Started log writer")
    while True:
        log_func = log_queue.get()
        log_func()
        log_queue.task_done()
        time.sleep(0.1)


logger.configure(
    handlers=[
        dict(
            sink=sys.stderr,
            level="INFO",
            backtrace=False,
            diagnose=False,
            enqueue=True,
        ),
        # dict(sink="logs.log", backtrace=True, diagnose=True, level='DEBUG', delay=True),
    ]
)


class SagesFreqtradeEnv4(gym.Env):
    """A freqtrade trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human", "system", "none"]}

    def __init__(
        self,
        data: dict[str, np.ndarray],
        prices: pd.DataFrame,
        window_size: int,
        pair: str,
        stake_amount: float,
        starting_balance: float,
        stop_loss=-0.25,
        punish_holding_amount=0,
        fee=0.005,
        writer: Optional[SummaryWriter] = None,
        optimal_duration=datetime.timedelta(days=3),
        optimal_duration_modifier=1,
    ):
        self.data = data
        self.window_size = window_size
        self.prices = prices

        self.pair = pair
        self.stake_amount = stake_amount
        self.stop_loss = stop_loss
        self.initial_balance = starting_balance
        self.current_balance = starting_balance
        self.price_column = "open"
        self.writer = writer
        assert self.stop_loss <= 0, "`stoploss` should be less or equal to 0"
        self.punish_holding_amount = punish_holding_amount
        assert (
            self.punish_holding_amount <= 0
        ), "`punish_holding_amount` should be less or equal to 0"
        self.fee = fee

        self.opened_trade: Optional[LocalTrade] = None
        self._trades: list[dict] = []

        self.current_episode = 0

        self.step_reward = 0
        self.total_reward = 0
        # if there is less than $10 in the account, we will stop trading
        self.min_balance_allowed = 10

        self.shape = list(data.values())[0].shape
        # self.shape = (self.window_size, number_of_features)
        self.action_space = spaces.Discrete(len(Actions))
        # self.action_space = spaces.MultiDiscrete([len(Actions), 10])
        shape = self.shape
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=shape, dtype=np.float64
        )
        logger.info(f"Observation space: {self.observation_space}")
        # self.observation_space = spaces.Box(
        #     low=-np.inf, high=np.inf, shape=shape, dtype=np.float32
        # )
        self._end_tick: int = 0
        self._current_tick: int = 0
        self._global_step = 0
        self.seed()

        # logger.info(f"Data columns: {data.columns.to_list()}, shape: {data.shape}")
        logger.info(
            f"Prices columns: {prices.columns.to_list()}, shape: {prices.shape}"
        )
        self.log_queue = Queue()

        # self.log_thread = Thread(target=log_writer, args=(self.log_queue,))
        # self.log_thread.start()

        self.log_dir = None
        self.buy_observation_map = {}
        self.sell_observation_map = {}

        self.dates = list(self.data.keys())

        self.optimal_duration = optimal_duration
        self.optimal_duration_modifier = optimal_duration_modifier

    # region Properties
    @property
    def _current_tick_date(self) -> Timestamp:
        return self.dates[self._current_tick]

    @property
    def _next_tick_date(self) -> Timestamp:
        return self.dates[self._current_tick + 1]

    @property
    def _prev_tick_date(self) -> Timestamp:
        return self.dates[self._current_tick - 1]

    @property
    def _current_rate(self):
        date_row = self.prices[self.prices["date"] == self._current_tick_date][
            self.price_column
        ]
        try:
            return date_row.iloc[0]
        except IndexError:
            logger.info(f"Date_row: {date_row}, tick date: {self._current_tick_date}")
            raise

    @property
    def _next_rate(self):
        return self.prices[self.prices["date"] == self._next_tick_date][
            self.price_column
        ].iloc[0]

    @property
    def _prev_rate(self):
        return self.prices[self.prices["date"] == self._prev_tick_date][
            self.price_column
        ].iloc[0]

    @property
    def _get_profit(self):
        if not self.opened_trade:
            return 0
        try:
            return self.opened_trade.calc_profit_ratio(self._current_rate, self.fee)
        except KeyError:
            logger.warning("Reached end of data")
            return self.opened_trade.calc_profit_ratio(self._prev_rate, self.fee)

    @property
    def balance_with_profit(self):
        return self.current_balance + self._get_profit

    @property
    def best_trade(self) -> Optional[LocalTrade]:
        """
        Find the trade with the highest reward
        :return: The best trade in terms of reward.
        """
        sells = [t for t in self.winning_trades]
        if len(sells) > 0:
            return max(sells, key=lambda t: self._calc_reward(t))

    @property
    def worst_trade(self):
        """
        Find the trade with the lowest reward
        :return: The worst trade in terms of reward.
        """
        sells = [t for t in self.losing_trades]
        if len(sells) > 0:
            return min(sells, key=lambda t: self._calc_reward(t))

    @property
    def closed_trades(self) -> list[LocalTrade]:
        """
        Return a list of all closed trades
        :return: A list of dictionaries.
        """
        return [t["trade"] for t in self._trades if t["type"] == "sell"]

    @property
    def winning_trades(self) -> list[LocalTrade]:
        """
        Return a list of all trades that were sold and had a positive profit
        :return: A list of trades that are winning trades.
        """
        return sorted(
            (t for t in self.closed_trades if self._calc_reward(t) > 0),
            key=lambda t: self._calc_reward(t),
        )

    @property
    def losing_trades(self):
        """
        Return a list of all trades that were closed with a loss
        :return: A list of LocalTrade objects.
        """
        return [
            t["trade"]
            for t in self._trades
            if t["type"] == "sell" and t["trade"].close_profit_abs < 0
        ]

    @property
    def trades_as_df(self):
        """Converts all closed trades to a dataframe"""
        trades_as_json = [t.to_json() for t in self.closed_trades]
        return pd.json_normalize(trades_as_json)

    @property
    def pre_numpy_observation(self):
        to_numpy = self.data[self._current_tick]

        return to_numpy

    # endregion
    def _get_observation(self):
        # data = to_numpy.drop(columns=["date"]).to_numpy()
        # has_open_trade = bool(self.opened_trade)
        return self.data[self._current_tick_date]

    def _take_action(self, action):
        # action, percent_of_balance = action
        # percent_of_balance = max(percent_of_balance, 1)
        if action == Actions.Hold.value:
            # the NN chose to hold

            # set the base reward to the punish_holding_amount
            self.step_reward = self.punish_holding_amount

            # is there an open trade?
            # if self.opened_trade:
            #     # what is the current profit?
            #     profit_percent = self.opened_trade.calc_profit_ratio(rate=self._current_rate)
            # is the profit below the stop loss?
            # if profit_percent <= self.stop_loss:
            #     # if it is...
            #     # set the reward to the current profit
            #     self._sell()

        if action == Actions.Buy.value:
            # the NN chose to buy

            if not self.opened_trade:
                # there is no trade open, so create a new one
                # stake_amount = percent_of_balance / 10 * self.current_balance * 0.99
                stake_amount = self.stake_amount
                try:
                    open_rate = self._next_rate
                    open_date = self._next_tick_date
                    tick = self._current_tick + 1
                except IndexError:
                    logger.info("Reached end of data, using last rate")
                    open_rate = self._current_rate
                    open_date = self._current_tick_date
                tick = self._current_tick
                self.opened_trade = LocalTrade(
                    id=len(self.closed_trades),
                    pair=self.pair,
                    open_rate=open_rate,
                    open_date=open_date,
                    stake_amount=stake_amount,
                    amount=stake_amount / open_rate,
                    fee_open=self.fee,
                    fee_close=self.fee,
                    is_open=True,
                )
                # record the trade
                self._trades.append(
                    {
                        "step": tick,
                        "type": "buy",
                        "total": open_rate,
                        "trade": self.opened_trade,
                    }
                )
                logger.debug(
                    f"[{open_date}] Buy @ {open_rate} | "
                    # f"POB: {percent_of_balance * 10}% | "
                    f"Stake: ${self.opened_trade.stake_amount:.2f} "
                )
                self.buy_observation_map[
                    (self.current_episode, len(self.closed_trades) + 1)
                ] = self._get_observation()

        elif action == Actions.Sell.value:
            # the NN chose to sell
            if self.opened_trade:
                # close the trade
                self._sell()

    def _sell(self):
        # try:
        # using next ticker because of freqtrade backtesting assumption:
        # Sell-signal sells happen at open-price of the *consecutive candle*
        # https://www.freqtrade.io/en/stable/backtesting/#assumptions-made-by-backtesting
        #     self.opened_trade.close_date = self._next_tick_date
        #     self.opened_trade.close(self._next_rate)
        #     tick = self._current_tick + 1
        # except KeyError:
        #     logger.warning(
        #         "Reached end of data... Closign trade with current rate and date"
        #     )
        try:
            close_date = self._next_tick_date
            close_rate = self._next_rate
        except IndexError:
            logger.warning(
                "Reached end of data... Closing trade with current rate and date"
            )
            close_date = self._current_tick_date
            close_rate = self._current_rate
        self.opened_trade.close_date = close_date
        self.opened_trade.close(close_rate)

        tick = self._current_tick

        self.step_reward = self._calc_reward(self.opened_trade)
        self.current_balance += self.opened_trade.close_profit_abs
        # record the trade
        self._trades.append(
            {
                "step": tick,
                "type": "sell",
                "total": close_rate,
                "trade": self.opened_trade,
            }
        )
        logger.debug(
            f"[{self.opened_trade.close_date}] Selling {self.opened_trade.amount:.8f} @ {self.opened_trade.close_rate} | "
            f"Profit: ${self.opened_trade.close_profit_abs:.2f} | "
            f"Profit %: {self.opened_trade.calc_profit_ratio(fee=self.fee):.3f}% | "
            f"Balance: ${self.current_balance:.2f}"
        )
        self.sell_observation_map[
            (self.current_episode, len(self.closed_trades))
        ] = self._get_observation()
        self.opened_trade = None

    def _calculate_duration_percent(self, trade: LocalTrade):
        trade_seconds = (trade.close_date - trade.open_date).total_seconds()
        optimal_seconds = self.optimal_duration.total_seconds()
        if trade_seconds < optimal_seconds or trade.close_profit < 0:
            return 1
        # calculate the percent increase from trade duration to optimal duration
        return (
            (trade_seconds - optimal_seconds) / optimal_seconds
        ) * self.optimal_duration_modifier

    def _calc_reward(self, trade: LocalTrade):
        """
        Calculate the reward for a trade

        :param trade: The trade object that we're calculating the reward for
        :return: The reward
        """
        # noinspection PyTypeChecker
        # trade_duration: datetime.timedelta = trade.close_date - trade.open_date
        ## reward quick profits and punish longer losses
        # if trade.close_profit_abs > 0:
        #     reward = trade.close_profit_abs / (trade_duration.total_seconds() / 3600)
        # else:
        #     reward = trade.close_profit_abs / (trade_duration.total_seconds() / 3600)

        # reward += reward * 0.1
        # set the reward to the profit per hour
        # this incentivizes shorter trades
        if trade.close_profit < 0:
            return trade.close_profit * 100

        # duration_percent = self._calculate_duration_percent(trade)
        initial_reward = trade.close_profit * 100
        # return max((initial_reward - (initial_reward * duration_percent), 0))
        return initial_reward

    def step(self, action):
        """
        Given an action, it will execute it and return the next observation, reward, and if the episode
        is done

        :param action: The action we took. This is what we will learn how to compute
        :return: The observation, the reward, whether the episode is done, and info.
        """
        # Execute one time step within the environment
        done = False

        self.step_reward = 0
        self._global_step += 1

        # are we at the end of the data or out of capital?
        if self._current_tick >= len(self.dates) - 2:
            # if so, it's time to stop
            done = True
            # if self.opened_trade:
            #     self._sell()
            # self.render(),.,.g
            self.log({"Step": self.get_info()})
            self.current_episode += 1

            # self.log_queue.put(self.log)
            # logger.info(f"End of data reached. Ending episode.")
        elif self.current_balance < self.stake_amount:
            # if we are out of capital, stop
            done = True
            # if self.opened_trade:
            #     self._sell()
            logger.info(f"Out of capital. Trades made: {len(self.closed_trades)}")
            # self.render()
            # self.log_queue.put(self.log)
            self.log({"Step": self.get_info()})
            self.current_episode += 1

        self._take_action(action)

        # proceed to the next day (candle)
        self._current_tick += 1
        self.total_reward += self.step_reward
        observation = self._get_observation()
        # print("Last ticker of observation: ", observation[-1])
        # print("Next ticker: ", self.data.iloc[self._current_tick + 1]["return"])

        return observation, self.step_reward, done, self.get_info().copy()

    def get_info(self):
        info = {
            # "step": self._current_tick,
            "total_reward": self.total_reward,
            "balance": self.current_balance,
            "trades": len(self.closed_trades),
            "total_profit_pct": round(get_total_profit_percent(self.closed_trades), 2),
            "custom_score": round(
                len(self.closed_trades) * get_total_profit_percent(self.closed_trades),
                2,
            ),
            "avg_profit_pct": round(get_average_ratio(self.closed_trades), 2),
            "winning_trades": len(self.winning_trades),
            "losing_trades": len(self.losing_trades),
            "win_ratio": round(
                calc_win_ratio(self.winning_trades, self.losing_trades), 2
            ),
            "average_duration (d)": get_average_trade_duration(
                self.closed_trades
            ).total_seconds()
            / 3600
            / 24,
            # "reward": self.step_reward,
        }

        return info

    def reset(self):
        """
        Reset the environment to an initial state
        :return: The observation space is a numpy array of size (window_size, data_dim)
        """
        # Reset the state of the environment to an initial state
        self.opened_trade = None
        self._trades = []

        self.step_reward = 0
        self.total_reward = 0
        self.current_balance = self.initial_balance
        self._current_tick = self.window_size + 1
        # self._end_tick = len(self.data) - 1

        return self._get_observation()

    def calculate_loss(self):
        """
        Calculate the average profit per trade and the win ratio
        :return: The average profit per trade, the win ratio, and the average profit ratio
        """
        # gather all sold trades
        sold_trades: list[LocalTrade] = [
            t["trade"] for t in self._trades if t["type"] == "sell"
        ]
        # turn them into a dataframe
        trades = [trade.to_json() for trade in sold_trades]
        results = pd.DataFrame(pd.json_normalize(trades))
        wins = len(results[results["profit_ratio"] > 0])
        avg_profit = results["profit_ratio"].sum() * 100.0
        win_ratio = wins / len(trades)
        return avg_profit * win_ratio * 100

    def set_log_dir(self, log_dir: Path):
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir
        logger.info(f"Log directory set to {log_dir}")

    def render(self, mode="human"):
        """
        The render function is called every time the environment is updated.
        used to print out the current state of the environment.
        The function is called with the mode parameter, which is used to specify if the environment is
        being rendered in human mode or agent mode.
        The agent mode is used to render the environment as a graph, which is useful for debugging

        :param mode: The mode of the environment. Can be either ‘human’ or ‘console’, defaults to human
        (optional)
        """
        if mode == "human":
            self.print()

    def log(self, info: dict = None):
        wandb_log = {}
        if not info:
            info = self.generate_stats()
        for name, stats in info.items():
            if not stats:
                continue
            for k, v in stats.items():
                key = f"{name}/{k}"

                if isinstance(v, datetime.timedelta):
                    v = v.total_seconds()
                    k = k + " (s)"
                else:
                    try:
                        v = float(str(v).strip("%$"))
                    except:
                        pass
                wandb_log[key] = v

        wandb.log(wandb_log)

        if self.current_episode % 20 == 0:
            if not any(self.closed_trades):
                return
            trades = self.trades_as_df
            trades.replace("", float("NaN"), inplace=True)
            trades.replace(0, float("NaN"), inplace=True)
            trades.dropna(how="all", axis=1, inplace=True)
            # make all columns lowercase
            trades.columns = [c.lower() for c in trades.columns]
            trades[
                [
                    "pair",
                    "stake_amount",
                    "amount",
                    "open_date",
                    "close_date",
                    "open_rate",
                    "close_rate",
                ]
            ].to_csv(self.log_dir / f"{self.current_episode}.csv", index=False)

    def generate_stats(self):
        stats = {
            "Tick": self._current_tick_date,
            "Trades": len(self.closed_trades),
            "Total reward": f"{self.total_reward:.4f}",
            "Average reward": f"{calc_average_reward(self.closed_trades, self._calc_reward):.4f}",
            "Balance": f"${self.current_balance:.2f}",
            "Total Profit pct": f"{get_total_profit_percent(self.closed_trades):.2f}%",
            "Avg profit pct": f"{get_average_ratio(self.closed_trades):.3f}%",
            "Avg duration": get_average_trade_duration(self.closed_trades),
            "Win Ratio": f"{calc_win_ratio(self.winning_trades, self.losing_trades):.3f}%",
        }

        best_trade__stats = (
            {
                "Tick": self.best_trade.open_date_utc,
                "Profit pct": f"{self.best_trade.calc_profit_ratio(fee=self.fee) * 100:.3f}%",
                "Open rate": f"{self.best_trade.open_rate:.5f}",
                "Close rate": f"{self.best_trade.close_rate:.5f}",
                "Duration": self.best_trade.close_date_utc
                - self.best_trade.open_date_utc,
                "Reward": f"{self._calc_reward(self.best_trade):.4f}",
            }
            if self.best_trade
            else {}
        )
        worst_trade__stats = (
            {
                "Tick": self.worst_trade.open_date_utc,
                "Profit pct": f"{self.worst_trade.calc_profit_ratio(fee=self.fee) * 100:.3f}%",
                "Open rate": f"{self.worst_trade.open_rate:.5f}",
                "Close rate": f"{self.worst_trade.close_rate:.5f}",
                "Duration": self.worst_trade.close_date_utc
                - self.worst_trade.open_date_utc,
                "Reward": f"{self._calc_reward(self.worst_trade):.4f}",
            }
            if self.worst_trade
            else {}
        )

        winning_trade__stats = (
            {
                "Count": len(self.winning_trades),
                "Total profit pct": f"{get_total_profit_percent(self.winning_trades):.2f}%",
                "Avg Profit %": f"{get_average_ratio(self.winning_trades):.3f}%",
                "Avg Duration": get_average_trade_duration(self.winning_trades),
                "Avg Reward": f"{calc_average_reward(self.winning_trades, self._calc_reward):.4f}",
                "Total Reward": f"{sum([self._calc_reward(t) for t in self.winning_trades]):.4f}",
            }
            if self.winning_trades
            else {}
        )
        losing_trade__stats = (
            {
                "Count": len(self.losing_trades),
                "Total profit pct": f"{get_total_profit_percent(self.losing_trades):.2f}%",
                "Avg Profit %": f"{get_average_ratio(self.losing_trades):.3f}%",
                "Avg Duration": get_average_trade_duration(self.losing_trades),
                "Avg Reward": f"{calc_average_reward(self.losing_trades, self._calc_reward):.4f}",
                "Total Reward": f"{sum([self._calc_reward(t) for t in self.losing_trades]):.4f}",
            }
            if self.losing_trades
            else {}
        )

        return {
            "General": stats,
            "Best Trade": best_trade__stats,
            "Worst Trade": worst_trade__stats,
            "Winning Trades": winning_trade__stats,
            "Losing Trades": losing_trade__stats,
        }

    def print(self):
        print("Stats".center(80, "-"))
        for name, stats in self.generate_stats().items():
            print(
                f"{name}:\n",
                " | ".join([f"{k}: {v}" for k, v in stats.items()]),
            )
            print()
        # self.writer.add_text(
        #     "Stats", " | ".join(string_append), global_step=self._global_step
        # )

        print("End of Stats".center(80, "-"))
        print()


register(
    # unique identifier for the env `name-version`
    id="MyFreqtradeEnv-v3",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point=SagesFreqtradeEnv4,
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=20000,
)
