import datetime
import json
import time
from datetime import timedelta
from pathlib import Path
from typing import Tuple

import mpu
import pandas as pd
import rapidjson
from freqtrade.configuration import Configuration, TimeRange
from freqtrade.data import history
from lazyft.command_parameters import HyperoptParameters
from lazyft.downloader import download_data_with_parameters
from lazyft.strategy import load_strategy
from loguru import logger
from pandas import DataFrame
from sb3_contrib.ppo_recurrent import RecurrentPPO
from sklearn.preprocessing import robust_scale
from stable_baselines3 import PPO
from stable_baselines3.a2c.a2c import A2C
from stable_baselines3.common.monitor import Monitor
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from tb_callbacks import SaveOnStepCallback
from trading_environments.my_freqtrade_env import SagesFreqtradeEnv

# region FT Settings
STRATEGY = 'FreqGymRScaler'
CONFIG = 'user_data/config.json'
PAIR = "BTC/USDT"
TRAINING_RANGE = "20210101-20211231"
TIMEFRAME = '15m'
freqtrade_config = Configuration.from_files([CONFIG])
freqtrade_config['timeframe'] = TIMEFRAME
freqtrade_config['pairs'] = [PAIR]
WINDOW_SIZE = 10
REQUIRED_STARTUP_CANDLES = 495
# endregion

# todo load last saved score and use it as a starting point to prevent overwriting
MODEL_NAME = ''

LOAD_PREPROCESSED_DATA = True  # useful if you have to calculate a lot of features
SAVE_PREPROCESSED_DATA = True

LEARNING_TIME_STEPS = 4000000
LOG_DIR = "./logs/"
TENSORBOARD_LOG = "./tensorboard/"
MODEL_DIR = Path("./models/")
_preprocessed_data_file = Path(
    'preprocessed',
    f"preprocessed_data__{PAIR.replace('/', '_')}__{TRAINING_RANGE}__{WINDOW_SIZE}.pickle",
)
"""End of settings"""
INDICATOR_FILTER = ['date', 'open', 'close', 'high', 'low', 'volume']

# hmm_model = Path(MODEL_DIR, f'btc_hmm.pickle')
model_dict_path = MODEL_DIR / 'models.json'
if not model_dict_path.exists():
    model_dict = {}
else:
    model_dict = rapidjson.loads(model_dict_path.read_text())


# def add_record(key: str, strategy: str, model_name: str, model_type: str, final: bool):
#     record = {
#         'strategy': strategy,
#         'pair': PAIR,
#         'timeframe': freqtrade_config.get('timeframe'),
#         'training_range': TRAINING_RANGE,
#         'save_name': model_name,
#         'model': model_type,
#         # 'reward': reward,
#     }
#     model_dict[key] = {'final' if final else 'best': record}
#     model_dict_path.write_text(rapidjson.dumps(model_dict, indent=4))


def _load_indicators(
    strategy_name: str = None, freqtrade_config: dict = None
) -> tuple[DataFrame, DataFrame]:
    """
    Loads the data, preprocesses it, and returns the indicators

    :param strategy_name: The name of the strategy to use
    :type strategy_name: str
    :param freqtrade_config: The configuration dictionary for the bot
    :type freqtrade_config: dict
    :return: the strategy and the indicators.
    """
    strategy = load_strategy(strategy_name, freqtrade_config)
    parameters = HyperoptParameters(
        timerange=TRAINING_RANGE,
        interval=TIMEFRAME,
        pairs=[PAIR],
        config_path=CONFIG,
    )
    download_data_with_parameters(strategy_name, parameters)
    if LOAD_PREPROCESSED_DATA:
        logger.info("Loading preprocessed data from file")
        assert _preprocessed_data_file.exists(), "Unable to load preprocessed data!"
        populated_data = mpu.io.read(str(_preprocessed_data_file))
        assert PAIR in populated_data, f"Loaded preprocessed data does not contain pair {PAIR}!"
        populated_pair_data = populated_data[PAIR]
    else:
        logger.info('Preprocessing data...')
        ohlc_data = _load_data(
            freqtrade_config, PAIR, TIMEFRAME, TRAINING_RANGE, REQUIRED_STARTUP_CANDLES
        )
        populated_data = strategy.advise_all_indicators(ohlc_data)
        populated_pair_data = populated_data[PAIR]
        populated_pair_data.reset_index(drop=True, inplace=True)
        logger.info('Dropping rows with NaN values')
        populated_pair_data.dropna(inplace=True)
        logger.info(f'Temp new index begins at: {populated_pair_data.index[0]}')
        populated_pair_data.reset_index(drop=True, inplace=True)
        # trading_env = FreqtradeEnv(
        #     data=pair_data,
        #     prices=price_data,
        #     window_size=WINDOW_SIZE,  # how many past candles should it use as features
        #     pair=PAIR,
        #     stake_amount=freqtrade_config['stake_amount'],
        #     punish_holding_amount=0,
        #     )

        # trading_env = SimpleROIEnv(
        #     data=pair_data,
        #     prices=price_data,
        #     window_size=WINDOW_SIZE,  # how many past candles should it use as features
        #     required_startup=required_startup,
        #     minimum_roi=0.02,  # 2% target ROI
        #     roi_candles=24,  # 24 candles * 5m = 120 minutes
        #     punish_holding_amount=0,
        #     punish_missed_buy=True
        #     )
        if SAVE_PREPROCESSED_DATA:
            logger.info("Saving preprocessed data to file")
            mpu.io.write(str(_preprocessed_data_file), {PAIR: populated_pair_data})

    price_data = populated_pair_data[INDICATOR_FILTER]
    # for c in INDICATOR_FILTER:
    #     # remove every column that contains a substring of c
    #     indicators = indicators.drop(columns=[col for col in indicators.columns if c in col])
    # indicators.fillna(0, inplace=True)
    input_data = price_data.drop(columns=['date', 'volume'])
    input_data = pd.DataFrame(
        robust_scale(input_data.values, quantile_range=(0.1, 100 - 0.1)),
        columns=input_data.columns,
        index=input_data.index,
    )
    return input_data, price_data


def _load_data(config, pair, timeframe, timerange, required_startup_candles):
    timerange = TimeRange.parse_timerange(timerange)

    return history.load_data(
        datadir=config['datadir'],
        pairs=[pair],
        timeframe=timeframe,
        timerange=timerange,
        startup_candles=required_startup_candles + 1,
        fail_without_data=True,
        data_format=config.get('dataformat_ohlcv', 'json'),
    )


def main():
    """
    Main function
    """
    indicators, price_data = _load_indicators(STRATEGY, freqtrade_config)

    env = SagesFreqtradeEnv(
        data=indicators,
        prices=price_data,
        window_size=WINDOW_SIZE,  # how many past candles should it use as features
        pair=PAIR,
        stake_amount=100,
        starting_balance=freqtrade_config['starting_balance'],
        punish_holding_amount=0,
    )
    # capitalize the first letter in every column name of price data
    # price_data.columns = [c.title() for c in price_data.columns]
    # env_maker = lambda: gym.make('stocks-v0', df=price_data, frame_bound=(5, 100), window_size=5)
    # env = DummyVecEnv([env_maker])

    trading_env = Monitor(env, LOG_DIR)

    # Optional policy_kwargs
    # see https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html?highlight=policy_kwargs#custom-network-architecture
    # policy_kwargs = dict(activation_fn=th.nn.ReLU,
    #                  net_arch=[dict(pi=[32, 32], vf=[32, 32])])
    # policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[32, dict(pi=[64,  64], vf=[64, 64])])
    # policy_kwargs = dict(net_arch=[32, dict(pi=[64, 64], vf=[64, 64])])
    policy_kwargs = dict(activation_fn=nn.ReLU)

    start_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if MODEL_NAME:
        # load existing model
        model = RecurrentPPO.load(
            MODEL_DIR / MODEL_NAME.strip('.zip'),
            tensorboard_log=TENSORBOARD_LOG,
        )
        logger.success(f'Loaded model from {MODEL_DIR / MODEL_NAME}')
        model.set_env(trading_env)
    else:
        # policy = RecurrentActorCriticPolicy
        model = RecurrentPPO(
            # See https://stable-baselines3.readthedocs.io/en/master/guide/algos.html for other algos with discrete action space
            "MlpLstmPolicy",
            trading_env,
            verbose=0,
            device='cuda',
            tensorboard_log=TENSORBOARD_LOG,
            n_steps=1024,
            # gamma=0.9391973108460121,
            # learning_rate=0.0001,
            # ent_coef=0.0001123894292050861,
            # gae_lambda=0.8789545362092943,
            # reuse=True
            policy_kwargs=policy_kwargs,
        )

    base_name = (
        f"{STRATEGY}_{trading_env.env.__class__.__name__}_{model.__class__.__name__}_{start_date}"
    )
    tb_callback = SaveOnStepCallback(
        check_freq=35000,
        save_name=f"best_model_{base_name}",
        save_dir=str(MODEL_DIR),
        log_dir=LOG_DIR,
        verbose=1,
    )
    # add_record(
    #     KEY,
    #     strategy.get_strategy_name(),
    #     f"best_model_{base_name}",
    #     model.__class__.__name__,
    #     final=False,
    # )
    logger.info(
        f"You can run tensorboard with: 'tensorboard --logdir {Path(TENSORBOARD_LOG).absolute()}'"
    )
    logger.info("Learning started.")

    t1 = time.time()
    tb_log_name = f'{model.__class__.__name__}_{STRATEGY}_{start_date}_cont={bool(MODEL_NAME)}'
    env.writer = SummaryWriter(log_dir=str(Path(TENSORBOARD_LOG, tb_log_name)))
    model.learn(
        total_timesteps=LEARNING_TIME_STEPS, callback=[tb_callback], tb_log_name=tb_log_name
    )
    print('Elapsed time:', timedelta(seconds=time.time() - t1))
    model.save(MODEL_DIR / f"final_model_{base_name}")
    # add_record(
    #     KEY,
    #     strategy.get_strategy_name(),
    #     f"final_model_{base_name}",
    #     model.__class__.__name__,
    #     final=True,
    # )


if __name__ == "__main__":
    main()
