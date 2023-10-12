from datetime import timedelta
from typing import Callable

from freqtrade.persistence import Trade
from gym.utils import seeding


def get_average_ratio(trades: list[Trade]) -> float:
    """
    Calculate the average profit of a list of trades
    :param trades: A list of trades
    :return: The average profit of the trades
    """
    if len(trades) == 0:
        return 0
    return sum([t.calc_profit_ratio(t.close_rate) for t in trades]) / len(trades) * 100


def get_total_profit_percent(trades: list[Trade]) -> float:
    """
    Calculate the total profit of a list of trades
    :param trades: A list of trades
    :return: The total profit of the trades
    """
    return sum([t.calc_profit_ratio(t.close_rate) for t in trades]) * 100


def get_average_stake(trades: list[Trade]) -> float:
    """
    Calculate the average profit of a list of trades
    :param trades: A list of trades
    :return: The average profit of the trades
    """
    if len(trades) == 0:
        return 0
    return sum([t.stake_amount for t in trades]) / len(trades)


def get_total_abs_profit(trades: list[Trade]) -> float:
    """
    Calculate the total profit of a list of trades
    :param trades: A list of trades
    :return: The total profit of the trades
    """
    return sum([t.close_profit_abs for t in trades])


def get_average_trade_duration(trades: list[Trade]) -> timedelta:
    """
    Calculate the average profit of a list of trades
    :param trades: A list of trades
    :return: The average profit of the trades
    """
    if len(trades) == 0:
        return timedelta(0)
    trades_to_delta = [t.close_date_utc - t.open_date_utc for t in trades]
    delta_to_seconds = [d.total_seconds() for d in trades_to_delta]
    average_seconds = sum(delta_to_seconds) / len(trades)
    seconds_to_delta = timedelta(seconds=average_seconds)
    return seconds_to_delta


def calc_average_reward(trades: list[Trade], reward_func: Callable[[Trade], float]) -> float:
    """
    Calculate the average profit of a list of trades
    :param trades: A list of trades
    :param reward_func: A function that takes a trade and returns a reward
    :return: The average profit of the trades
    """
    if len(trades) == 0:
        return 0
    return sum([reward_func(t) for t in trades]) / len(trades)


def calc_win_ratio(winning_trades: list[Trade], losing_trades: list[Trade]) -> float:
    """
    Return the win ratio of the trades

    :param winning_trades: A list of winning trades
    :param losing_trades: A list of losing trades
    :return: The win ratio is being returned.
    """
    if not winning_trades and not losing_trades:
        return 0
    return len(winning_trades) / (len(winning_trades) + len(losing_trades)) * 100


def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]
