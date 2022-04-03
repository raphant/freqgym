import datetime
import json
from enum import Enum
from typing import Optional

import gym
import numpy as np
import pandas as pd
from freqtrade.persistence import Trade
from gym import spaces
from loguru import logger

# logger.configure(
#     handlers=[
#         dict(sink=sys.stderr, level='DEBUG', backtrace=False, diagnose=False, enqueue=True),
#         # dict(sink="logs.log", backtrace=True, diagnose=True, level='DEBUG', delay=True),
#     ]
# )
from pandas import Timestamp

# Based on https://github.com/hugocen/freqtrade-gym/blob/master/freqtradegym.py
from torch.utils.tensorboard import SummaryWriter

from trading_environments.util import (
    calc_average_reward,
    calc_win_ratio,
    get_average_ratio,
    get_average_stake,
    get_average_trade_duration,
    get_total_profit_percent,
)

# create an Dic


class Actions(Enum):
    # Hold = 0
    Buy = 0
    Sell = 1


class SagesFreqtradeEnv(gym.Env):
    """A freqtrade trading environment for OpenAI gym"""

    metadata = {'render.modes': ['human', 'system', 'none']}

    def __init__(
        self,
        data: pd.DataFrame,
        prices: pd.DataFrame,
        window_size: int,
        pair: str,
        stake_amount: float,
        starting_balance: float,
        stop_loss=-0.25,
        punish_holding_amount=0,
        fee=0.005,
        writer: Optional[SummaryWriter] = None,
    ):
        self.data = data
        self.window_size = window_size
        self.prices = prices

        self.pair = pair
        self.stake_amount = stake_amount
        self.stop_loss = stop_loss
        self.initial_balance = starting_balance
        self.current_balance = starting_balance
        self.price_column = 'open'
        self.writer = writer
        assert self.stop_loss <= 0, "`stoploss` should be less or equal to 0"
        self.punish_holding_amount = punish_holding_amount
        assert (
            self.punish_holding_amount <= 0
        ), "`punish_holding_amount` should be less or equal to 0"
        self.fee = fee

        self.opened_trade: Optional[Trade] = None
        self._trades: list[dict] = []

        self.step_reward = 0
        self.total_reward = 0
        # if there is less than $10 in the account, we will stop trading
        self.min_balance_allowed = 10

        _, number_of_features = self.data.shape
        self.shape = (self.window_size, number_of_features)
        self.action_space = spaces.Discrete(len(Actions))
        # self.action_space = spaces.MultiDiscrete([len(Actions), 10])
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32
        )
        self._end_tick: int = 0
        self._current_tick: int = 0
        self._global_step = 0
        self.seed()

        logger.info(f"Data columns: {data.columns.to_list()}, shape: {data.shape}")
        logger.info(f'Prices columns: {prices.columns.to_list()}, shape: {prices.shape}')

    @property
    def _current_tick_date(self) -> Timestamp:
        return self.prices.loc[self._current_tick]['date']

    @property
    def _current_rate(self):
        return self.prices.loc[self._current_tick][self.price_column]

    # @property
    # def trades(self):
    #     trades = self.closed_trades
    #     return

    @property
    def best_trade(self) -> Optional[Trade]:
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
    def closed_trades(self) -> list[Trade]:
        """
        Return a list of all closed trades
        :return: A list of dictionaries.
        """
        return [t['trade'] for t in self._trades if t['type'] == 'sell']

    @property
    def winning_trades(self) -> list[Trade]:
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
        :return: A list of Trade objects.
        """
        return [
            t['trade']
            for t in self._trades
            if t['type'] == 'sell' and t['trade'].close_profit_abs < 0
        ]

    def _get_observation(self):
        return self.data[(self._current_tick - self.window_size) : self._current_tick].to_numpy()

    def _take_action(self, action):
        # action, percent_of_balance = action
        # percent_of_balance = max(percent_of_balance, 1)
        # if action == Actions.Hold.value:
        #     # the NN chose to hold
        #
        #     # set the base reward to the punish_holding_amount
        #     self.step_reward = self.punish_holding_amount
        #
        #     # is there an open trade?
        #     if self.opened_trade:
        #         # what is the current profit?
        #         profit_percent = self.opened_trade.calc_profit_ratio(rate=self._current_rate)
        #         # is the profit below the stop loss?
        #         # if profit_percent <= self.stop_loss:
        #         #     # if it is...
        #         #     # set the reward to the current profit
        #         #     self._sell()

        if action == Actions.Buy.value:
            # the NN chose to buy

            if not self.opened_trade:
                # there is no trade open, so create a new one
                # stake_amount = percent_of_balance / 10 * self.current_balance * 0.99
                stake_amount = self.stake_amount
                self.opened_trade = Trade(
                    pair=self.pair,
                    open_rate=self._current_rate,
                    open_date=self._current_tick_date,
                    stake_amount=stake_amount,
                    amount=stake_amount / self._current_rate,
                    fee_open=self.fee,
                    fee_close=self.fee,
                    is_open=True,
                )
                # record the trade
                self._trades.append(
                    {
                        "step": self._current_tick,
                        "type": 'buy',
                        "total": self._current_rate,
                        "trade": self.opened_trade,
                    }
                )
                logger.debug(
                    f"[{self._current_tick_date}] Buy @ {self._current_rate} | "
                    # f"POB: {percent_of_balance * 10}% | "
                    f"Stake: ${self.opened_trade.stake_amount:.2f} "
                )

        elif action == Actions.Sell.value:
            # the NN chose to sell
            if self.opened_trade:
                # close the trade
                self._sell()

    def _sell(self):
        self.opened_trade.close_date = self._current_tick_date
        self.opened_trade.close(self._current_rate)
        self.step_reward = self._calc_reward(self.opened_trade)
        self.current_balance += self.opened_trade.close_profit_abs
        # record the trade
        self._trades.append(
            {
                "step": self._current_tick,
                "type": 'sell',
                "total": self._current_rate,
                "trade": self.opened_trade,
            }
        )
        logger.debug(
            f"[{self._current_tick_date}] Selling {self.opened_trade.amount:.8f} @ {self._current_rate} | "
            f"Profit: ${self.opened_trade.close_profit_abs:.2f} | "
            f"Profit %: {self.opened_trade.calc_profit_ratio(fee=self.fee):.3f}% | "
            f"Balance: ${self.current_balance:.2f}"
        )
        self.opened_trade = None

    @staticmethod
    def _calc_reward(trade: Trade):
        """
        Calculate the reward for a trade based on the profit and the duration of the trade

        :param trade: The trade object that we're calculating the reward for
        :return: The reward is the profit per hour.
        """
        # noinspection PyTypeChecker
        trade_duration: datetime.timedelta = trade.close_date - trade.open_date
        # reward quick profits and punish longer losses
        if trade.close_profit_abs > 0:
            reward = trade.close_profit_abs / (trade_duration.total_seconds() / 3600)
        else:
            reward = trade.close_profit_abs / (trade_duration.total_seconds() / 3600)
            # reward += reward * 0.1
        # set the reward to the profit per hour
        # this incentivizes shorter trades
        return reward

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
        if self._current_tick >= self._end_tick:
            # if so, it's time to stop
            done = True
            if self.opened_trade:
                self._sell()
            self.render()
            # logger.info(f"End of data reached. Ending episode.")
        elif self.current_balance < self.stake_amount:
            # if we are out of capital, stop
            done = True
            if self.opened_trade:
                self._sell()
            logger.info(f"Out of capital. Trades made: {len(self.closed_trades)}")
            self.render()

        self._take_action(action)

        # proceed to the next day (candle)
        self._current_tick += 1
        self.total_reward += self.step_reward
        observation = self._get_observation()
        info = {
            "step": self._current_tick,
            "total_reward": self.total_reward,
            "balance": self.current_balance,
            "trade": self.opened_trade,
        }
        return observation, self.step_reward, done, info

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
        self._end_tick = len(self.data) - 1

        return self._get_observation()

    def calculate_loss(self):
        """
        Calculate the average profit per trade and the win ratio
        :return: The average profit per trade, the win ratio, and the average profit ratio
        """
        # gather all sold trades
        sold_trades: list[Trade] = [t['trade'] for t in self._trades if t['type'] == 'sell']
        # turn them into a dataframe
        trades = [trade.to_json() for trade in sold_trades]
        results = pd.DataFrame(pd.json_normalize(trades))
        wins = len(results[results['profit_ratio'] > 0])
        avg_profit = results['profit_ratio'].sum() * 100.0
        win_ratio = wins / len(trades)
        return avg_profit * win_ratio * 100

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
            print('Stats'.center(80, '-'))
            string_append = [
                f"Tick: {self._current_tick_date}",
                f"Trades: {len(self.closed_trades)}",
                f"Total reward: {self.total_reward:.4f}",
                f'Average reward: {calc_average_reward(self.closed_trades, self._calc_reward):.4f}',
                f"Balance: ${self.current_balance:.2f}",
                # f"Total Profit: ${get_total_abs_profit(self.closed_trades):.2f}",
                f"Total Profit pct: {get_total_profit_percent(self.closed_trades):.2f}%",
                f"Avg profit pct: {get_average_ratio(self.closed_trades):.3f}%",
                # f"Avg Stake: ${get_average_stake(self.closed_trades):.2f}",
                f"Avg duration: {get_average_trade_duration(self.closed_trades)}",
                f"Win Ratio: {calc_win_ratio(self.winning_trades, self.losing_trades):.3f}%",
            ]
            print(" | ".join(string_append))
            self.writer.add_text("Stats", " | ".join(string_append), global_step=self._global_step)
            string_append.clear()
            print()
            # print the current state
            if self.opened_trade:
                string_append.extend(
                    [
                        f"Tick: {self.opened_trade.open_date}",
                        # f"Current Profit: ${self.opened_trade.close_profit_abs:.2f}",
                        f"\nProfit pct: {self.opened_trade.calc_profit_ratio(self._current_rate, fee=self.fee):.3f}%",
                        # f"Profit Abs: ${self.opened_trade.calc_profit(self._current_rate):.2f}",
                        # f"Amount: {self.opened_trade.amount:.5f}",
                        # f"Stake: ${self.opened_trade.stake_amount:.2f}",
                        f"Time open: {self._current_tick_date.to_pydatetime()-self.opened_trade.open_date}",
                    ]
                )
                print('Open Trade:', " | ".join(string_append))
                string_append.clear()
                print()

            if self.best_trade:
                string_append.extend(
                    [
                        f"\nTick: {self.best_trade.open_date_utc}",
                        f"Profit pct: {self.best_trade.calc_profit_ratio(fee=self.fee) * 100:.3f}%",
                        f"Open rate: {self.best_trade.open_rate:.5f}",
                        f"Close rate: {self.best_trade.close_rate:.5f}",
                        f"Duration: {self.best_trade.close_date_utc-self.best_trade.open_date_utc}",
                        f'Reward: {self._calc_reward(self.best_trade):.4f}',
                    ]
                )
                print('Best Trade:', ' | '.join(string_append))
                print()
                string_append.clear()
            if self.worst_trade:
                string_append.extend(
                    [
                        f"\nTick: {self.worst_trade.open_date_utc}",
                        f"Profit pct: {self.worst_trade.calc_profit_ratio(fee=self.fee) * 100:.3f}%",
                        f"Open rate: {self.worst_trade.open_rate:.5f}",
                        f"Close rate: {self.worst_trade.close_rate:.5f}",
                        f"Duration: {self.worst_trade.close_date_utc-self.worst_trade.open_date_utc}",
                        f'Reward: {self._calc_reward(self.worst_trade):.4f}',
                    ]
                )
                print('Worst Trade:', ' | '.join(string_append))
                print()
                string_append.clear()
            if any(self.winning_trades):
                string_append.extend(
                    [
                        f"\n\tNumber: {len(self.winning_trades)}",
                        f'Total profit pct: {get_total_profit_percent(self.winning_trades):.2f}%',
                        f"Avg Profit %: {get_average_ratio(self.winning_trades):.3f}%",
                        f"Avg Duration: {get_average_trade_duration(self.winning_trades)}",
                        f'Avg Reward: {calc_average_reward(self.winning_trades, self._calc_reward):.4f}',
                        f'Total Reward: {sum([self._calc_reward(t) for t in self.winning_trades]):.4f}',
                    ]
                )
                print('Winning Trades:', ' | '.join(string_append))
                print()
                string_append.clear()
            if any(self.losing_trades):
                string_append.extend(
                    [
                        f"\n\tNumber: {len(self.losing_trades)}",
                        f'Total profit pct: {get_total_profit_percent(self.losing_trades):.2f}%',
                        f"Avg Profit %: {get_average_ratio(self.losing_trades):.3f}%",
                        f"Avg Duration: {get_average_trade_duration(self.losing_trades)}",
                        f'Avg Reward: {calc_average_reward(self.losing_trades, self._calc_reward):.4f}',
                        f'Total Reward: {sum([self._calc_reward(t) for t in self.losing_trades]):.4f}',
                    ]
                )
                print('Losing Trades:', ' | '.join(string_append))
                print()
                string_append.clear()
            print('End of Stats'.center(80, '-'))
            print()
