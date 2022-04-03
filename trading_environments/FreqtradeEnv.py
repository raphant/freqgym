from enum import Enum

import gym
import numpy as np
from freqtrade.persistence import Trade
from gym import spaces
from gym.utils import seeding
from loguru import logger

# Based on https://github.com/hugocen/freqtrade-gym/blob/master/freqtradegym.py


class Actions(Enum):
    Hold = 0
    Buy = 1
    Sell = 2


class FreqtradeEnv(gym.Env):
    """A freqtrade trading environment for OpenAI gym"""

    metadata = {'render.modes': ['human', 'system', 'none']}

    def __init__(
        self,
        data,
        prices,
        window_size,
        pair,
        stake_amount,
        stop_loss=-0.15,
        punish_holding_amount=0,
        fee=0.005,
    ):
        self.data = data
        self.window_size = window_size
        self.prices = prices
        self.pair = pair
        self.stake_amount = stake_amount
        self.stop_loss = stop_loss
        self.price_value = 'open'
        assert self.stop_loss <= 0, "`stoploss` should be less or equal to 0"
        self.punish_holding_amount = punish_holding_amount
        assert (
            self.punish_holding_amount <= 0
        ), "`punish_holding_amount` should be less or equal to 0"
        self.fee = fee

        self.opened_trade = None
        self.trades = []

        self._reward = 0
        self.total_reward = 0

        _, number_of_features = self.data.shape
        self.shape = (self.window_size, number_of_features)
        # logger.info(f"Shape of observation: {self.shape}")
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32
        )

        self.seed()

    def _get_observation(self):
        return self.data[(self._current_tick - self.window_size) : self._current_tick].to_numpy()

    def _take_action(self, action):
        if action == Actions.Hold.value:
            # the NN chose to hold

            # set the base reward to the punish_holding_amount
            self._reward = self.punish_holding_amount

            # is there an open trade?
            if self.opened_trade:
                # what is the current profit?
                profit_percent = self.opened_trade.calc_profit_ratio(
                    rate=self.prices.loc[self._current_tick][self.price_value]
                )
                # is the profit below the stop loss?
                if profit_percent <= self.stop_loss:
                    # if it is...
                    # set the reward to the current profit
                    self._reward = profit_percent
                    # and close the trade
                    self.opened_trade = None

        elif action == Actions.Buy.value:
            # the NN chose to buy

            if not self.opened_trade:
                # there is no trade open, so create a new one
                self.opened_trade = Trade(
                    pair=self.pair,
                    open_rate=self.prices.loc[self._current_tick][self.price_value],
                    open_date=self.prices.loc[self._current_tick].date,
                    stake_amount=self.stake_amount,
                    amount=self.stake_amount
                    / self.prices.loc[self._current_tick][self.price_value],
                    fee_open=self.fee,
                    fee_close=self.fee,
                    is_open=True,
                )
                # record the trade
                self.trades.append(
                    {
                        "step": self._current_tick,
                        "type": 'buy',
                        "total": self.prices.loc[self._current_tick][self.price_value],
                    }
                )

        elif action == Actions.Sell.value:
            # the NN chose to sell
            if self.opened_trade:
                # what is the current profit?
                profit_percent = self.opened_trade.calc_profit_ratio(
                    rate=self.prices.loc[self._current_tick][self.price_value]
                )
                # close the trade
                self.opened_trade = None

                # set the reward to the current profit
                self._reward = profit_percent

                # record the trade
                self.trades.append(
                    {
                        "step": self._current_tick,
                        "type": 'sell',
                        "total": self.prices.loc[self._current_tick][self.price_value],
                    }
                )

    def step(self, action):
        # Execute one time step within the environment
        done = False

        self._reward = 0

        # are we at the end of the data?
        if self._current_tick >= self._end_tick:
            # if so, it's time to stop
            done = True

        self._take_action(action)

        # proceed to the next day (candle)
        self._current_tick += 1

        self.total_reward += self._reward

        observation = self._get_observation()

        return observation, self._reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.opened_trade = None
        self.trades = []

        self._reward = 0
        self.total_reward = 0

        self._current_tick = self.window_size + 1
        self._end_tick = len(self.data) - 1

        return self._get_observation()


def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]
