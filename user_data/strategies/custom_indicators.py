import numpy as np
import pandas as pd
import pandas_ta
import talib

import pandas_ta as ta
import finta as TA


def add_adx_di_signal(df, period=14):
    # Calculate ADX and DI
    adx = talib.ADX(df["high"], df["low"], df["close"], timeperiod=period)
    plus_di = talib.PLUS_DI(df["high"], df["low"], df["close"], timeperiod=period)
    minus_di = talib.MINUS_DI(df["high"], df["low"], df["close"], timeperiod=period)

    # Initialize signal column with 0
    df["adx_di_signal"] = 0

    # Loop through the dataframe
    for i in range(period, len(df)):
        # Check if ADX is above 25 and +DI is above -DI
        if adx[i] > 25 and plus_di[i] > minus_di[i]:
            df.loc[df.index[i], "adx_di_signal"] = 1
    return df


def calculate_smi(df, k_length, d_length):
    ll = df["low"].rolling(window=k_length).min()
    hh = df["high"].rolling(window=k_length).max()
    diff = hh - ll
    rdiff = df["close"] - (hh + ll) / 2
    avgrel = talib.EMA(talib.EMA(rdiff, d_length), d_length)
    avgdiff = talib.EMA(talib.EMA(diff, d_length), d_length)
    smi = np.where(avgdiff != 0, (avgrel / (avgdiff / 2) * 100), 0)
    return smi


def add_smi_signal(df, k_length=10, d_length=3, ema_length=10) -> pd.DataFrame:
    # Calculate SMI and EMA of SMI
    smi = calculate_smi(df, k_length, d_length)
    smi_ema = talib.EMA(smi, ema_length)

    # Initialize signal column with 0
    df["smi_signal"] = 0

    # Loop through the dataframe
    for i in range(2, len(df)):
        # Check if SMI crossed above lower band and crossed above the SMI EMA within the next 3 tickers
        if smi[i - 2] < -40 < smi[i - 1] and smi[i - 1] > smi_ema[i - 1]:
            df.loc[df.index[i], "smi_signal"] = 1
    return df


def add_ema_signal(df, fast_length=50, slow_length=200) -> pd.DataFrame:
    # Calculate fast and slow EMA
    fast_ema = talib.EMA(df["close"], fast_length)
    slow_ema = talib.EMA(df["close"], slow_length)

    # Initialize signal column with 0
    df["ema_signal"] = 0

    # Loop through the dataframe
    for i in range(1, len(df)):
        # Check if fast EMA > slow EMA and price touched, but closed above the fast EMA on last candle
        if (
            slow_ema[i - 1] < fast_ema[i - 1] < df.loc[df.index[i - 1], "close"]
            and df.loc[df.index[i - 1], "low"] <= fast_ema[i - 1]
        ):
            df.loc[df.index[i], "ema_signal"] = 1
    return df


def add_sar_signal(df, ema_length=200, window_length=200):
    # Calculate SAR and EMA
    sar = talib.SAR(df["high"], df["low"])
    ema = talib.EMA(df["close"], ema_length)

    # Calculate the distance between SAR and price
    sar_distance = abs(df["close"] - sar) / df["close"]

    # Initialize signal column with 0.0
    df["sar_signal"] = 0.0

    # Loop through the dataframe
    for i in range(window_length, len(df)):
        # Normalize sar_distance to range 0-1 using a rolling window
        min_sar_distance = min(
            sar_distance[i - window_length : i + 1].min(), sar_distance[i]
        )
        max_sar_distance = max(
            sar_distance[i - window_length : i + 1].max(), sar_distance[i]
        )
        normalized_sar_distance = (sar_distance[i] - min_sar_distance) / (
            max_sar_distance - min_sar_distance + 1e-10
        )

        # Check if SAR and price are above EMA200 and SAR crosses under price
        if (
            sar[i - 1] > ema[i - 1]
            and df.loc[df.index[i - 1], "close"] > ema[i - 1]
            and sar[i] < df.loc[df.index[i], "close"]
        ):
            # Check if SAR is diverging
            if (
                sar[i] < sar[i - 1]
                and df.loc[df.index[i], "close"] > df.loc[df.index[i - 1], "close"]
            ):
                # Set sar_signal to normalized sar_distance
                df.loc[df.index[i], "sar_signal"] = normalized_sar_distance
    return df


def calculate_smoothed_ha(df, smooth_length=5):
    # Calculate Heiken Ashi candles
    ha = pandas_ta.ha(df["open"], df["high"], df["low"], df["close"])

    # Smooth the Heiken Ashi candles
    ha_smooth = ha.ewm(span=smooth_length).mean()

    return ha_smooth


def add_ha_signal(df, smooth_length=5):
    # Calculate smoothed Heiken Ashi candles
    ha_smooth = calculate_smoothed_ha(df, smooth_length)

    # Initialize signal column with 0.0
    df["ha_signal"] = 0.0

    # Calculate the width of the Heiken Ashi bar
    ha_width = abs(ha_smooth["HA_close"] - ha_smooth["HA_open"])

    # Normalize ha_width to range 0-1
    min_ha_width = ha_width.min()
    max_ha_width = ha_width.max()
    normalized_ha_width = (ha_width - min_ha_width) / (max_ha_width - min_ha_width)

    # Loop through the dataframe
    for i in range(1, len(df)):
        # Check if HA is trending "green" or bullish
        if (
            ha_smooth.loc[ha_smooth.index[i], "HA_close"]
            > ha_smooth.loc[ha_smooth.index[i], "HA_open"]
        ):
            # Set ha_signal to normalized ha_width
            df.loc[df.index[i], "ha_signal"] = normalized_ha_width[i]
    return df


import numpy as np


def calculate_donchian_channel(df, period=200):
    # Calculate upper and lower band of the Donchian Channel
    upper_band = df["high"].rolling(window=period).max()
    lower_band = df["low"].rolling(window=period).min()

    return upper_band, lower_band


import numpy as np


def calculate_donchian_channel(df, period=200):
    # Calculate upper and lower band of the Donchian Channel
    upper_band = df["high"].rolling(window=period).max()
    lower_band = df["low"].rolling(window=period).min()

    return upper_band, lower_band


def add_donchian_trend_signal(df, period=200) -> pd.DataFrame:
    # Calculate Donchian Channel
    upper_band, lower_band = calculate_donchian_channel(df, period)

    # Initialize signal column with 0
    df["donchian_trend_signal"] = 0

    # Calculate the percentage that the price is above the bullish line for each candle
    df["percentage_above_bullish_line"] = (df["close"] - upper_band) / upper_band * 100

    # Calculate the average and standard deviation of these percentages over the period
    df["average_percentage"] = (
        df["percentage_above_bullish_line"].rolling(window=period).mean()
    )
    df["std_dev_percentage"] = (
        df["percentage_above_bullish_line"].rolling(window=period).std()
    )

    # Normalize the average percentage by dividing it by the sum of the average percentage and the standard deviation
    # Then apply the sigmoid function to squash the values between 0 and 1
    df["donchian_trend_signal"] = 1 / (
        1
        + np.exp(
            -df["average_percentage"]
            / (df["average_percentage"] + df["std_dev_percentage"])
        )
    )

    # Replace any NaN values with 0
    df["donchian_trend_signal"].fillna(0, inplace=True)

    # Drop the intermediate columns used for calculation
    df.drop(
        columns=[
            "percentage_above_bullish_line",
            "average_percentage",
            "std_dev_percentage",
        ],
        inplace=True,
    )

    return df


def add_rsi_signal(df: pd.DataFrame, period=14) -> pd.DataFrame:
    # Calculate RSI
    rsi = talib.RSI(df["close"], timeperiod=period)

    # Initialize signal column with 0
    df["rsi_signal"] = 0

    # Loop through the dataframe and check if RSI crossed above 30
    for i in range(period, len(df)):
        # Check if RSI is oversold
        if rsi[i] < 30:
            df.loc[df.index[i], "rsi_signal"] = 1
    return df


def add_bollinger_signal(df: pd.DataFrame, period=20, std_dev=2) -> pd.DataFrame:
    # Calculate Bollinger Bands
    upper_band, middle_band, lower_band = talib.BBANDS(
        df["close"], timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev
    )

    # Initialize signal column with 0
    df["bollinger_signal"] = 0

    # Loop through the dataframe and check if price is below lower band
    for i in range(len(df)):
        if df["close"].iloc[i] < lower_band.iloc[i]:
            df["bollinger_signal"].iloc[i] = 1
    return df
