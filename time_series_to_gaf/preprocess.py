import os
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path
from typing import Any, BinaryIO, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from diskcache import Cache
from keras_preprocessing.image import ImageDataGenerator, img_to_array
from lazyft import paths
from lazyft.data_loader import load_pair_data
from numpy import ndarray
from PIL.Image import Image
from torchvision import datasets
from torchvision.transforms import transforms

# add parent folder to path
from time_series_to_gaf.constants import TRAIN_IMAGES_PATH
from time_series_to_gaf.gafs import create_gaf, create_images, get_image_from_gaf

cache = Cache(paths.CACHE_DIR)


def data_to_image_preprocess(
    timerange='20180101-20211231',
    data: pd.DataFrame = None,
    pair: str = 'BTC/USDT',
    interval: str = '1h',
    image_save_path: Path = TRAIN_IMAGES_PATH,
):
    """
    This function takes a timerange and a dataframe and creates images for each timeframe.
    If no dataframe is passed, it will load the dataframe from the lazyft data.

    :param timerange: The time range to use for the data, defaults to 20180101-20211231 (optional)
    :param data: The dataframe to be converted to an image
    :param pair: The coin to be used for the data, defaults to 'BTC/USDT' (optional)
    :param interval: The interval to be used for the data, defaults to '30m' (optional)
    :param image_save_path: The path to save the images, defaults to IMAGES_PATH (optional)
    :type data: pd.DataFrame
    :return: A dataframe with the following columns:
        date, open, close, high, low, volume
    """
    df = data
    if data is None:
        df = load_pair_data(pair, interval, timerange=timerange)
    df = (
        df.drop(columns=['high', 'low', 'volume'], axis=1, errors='ignore')
        .groupby(pd.Grouper(key='date', freq='1h'))
        .mean()
        .reset_index()
    )
    # clean_df = clean_non_trading_times(df)
    return set_gaf_data(df, image_save_path=image_save_path)


def create_answers(dates: list[str], df: pd.DataFrame) -> list[int]:
    days = (
        df.drop(columns=['high', 'low', 'volume'], axis=1, errors='ignore')
        .groupby(pd.Grouper(key='date', freq='1d'))
        .mean()
        .reset_index()
    )
    # if close is greater than previous close, then set answer to 1
    # if close is less than previous close, then set answer to 0
    # if close is equal to previous close, then set answer to 0
    return days['close'].gt(days['close'].shift(1)).astype(int).fillna(0).tolist()


def quick_gaf(df: pd.DataFrame):
    df = (
        df.drop(columns=['high', 'low', 'volume'], axis=1, errors='ignore')
        .groupby(pd.Grouper(key='date', freq='1h'))
        .mean()
        .reset_index()
    )
    timeframes = ['1h', '2h', '4h', '1d']
    dates = df['date'].dt.date
    dates = dates.drop_duplicates()
    list_dates = dates.apply(str).tolist()

    gafs = split_timeframes(df, window=20, timeframes=timeframes, dates=list_dates)
    answers = create_answers(dates, df)
    return [series_quadrant_to_pil(gaf) for gaf in gafs], answers


def set_gaf_data(
    df: pd.DataFrame, window: int = 20, timeframes: list = None, image_save_path=TRAIN_IMAGES_PATH
):
    """
    It takes a dataframe of historical data and a window size,
    and generates a set of images for each trading decision (long or short) that occurred during that
    window

    :param df: The dataframe that contains the data
    :param window: The number of days to look back when calculating the GAF, defaults to 20
    :param timeframes: list of timeframes to use for the GAF
    :param image_save_path: The path to save the images, defaults to IMAGES_PATH (optional)
    """
    if timeframes is None:
        timeframes = ['1h', '4h', '8h', '1d']
    dates = df['date'].dt.date
    dates = dates.drop_duplicates()
    list_dates = dates.apply(str).tolist()

    index = window
    decision_map = defaultdict(list)

    while True:
        if index >= len(list_dates):
            break
        # select appropriate timeframe
        data_slice = df.loc[
            (df['date'] > list_dates[index - window]) & (df['date'] < list_dates[index])
        ]
        # print('len of data slice: ', len(data_slice), 'head: ', data_slice.head())
        gafs = []

        # group dataslices by timeframe
        for freq in timeframes:
            group_dt = data_slice.groupby(pd.Grouper(key='date', freq=freq)).mean().reset_index()
            group_dt = group_dt.dropna()
            gafs.append(group_dt['close'].tail(20))
        future_value = df.loc[df['date'] == list_dates[index]]['close'].iloc[-1]
        current_value = data_slice['close'].iloc[-1]
        decision = trading_action(future_value, current_value)
        decision_map[decision].append([list_dates[index - 1], gafs])
        index += 1
    print('Generating images...')
    generate_gaf(decision_map, image_save_path)
    dt_points = dates.shape[0]
    total_shorts = len(decision_map['SHORT'])
    total_longs = len(decision_map['LONG'])
    images_created = total_shorts + total_longs
    print(
        "========PREPROCESS REPORT========:\nTotal Data Points: {0}\nTotal Images Created: {1}"
        "\nTotal LONG positions: {2}\nTotal SHORT positions: {3}".format(
            dt_points, images_created, total_shorts, total_longs
        )
    )


def split_timeframes(df, window, timeframes, dates: list[str]):
    index = window
    gafs_list = []
    while True:
        if index >= len(dates):
            break
        # select appropriate timeframe
        data_slice = df.loc[(df['date'] > dates[index - window]) & (df['date'] < dates[index])]
        # print('len of data slice: ', len(data_slice), 'head: ', data_slice.head())
        gafs: list[pd.Series] = []

        # group dataslices by timeframe
        for freq in timeframes:
            group_dt: pd.DataFrame = (
                data_slice.groupby(pd.Grouper(key='date', freq=freq)).mean().reset_index()
            )
            group_dt = group_dt.dropna()
            gafs.append(group_dt['close'].tail(20))
        gafs_list.append(gafs)
        index += 1
    return gafs_list


def generate_gaf(images_data: dict[str, list], image_save_path=TRAIN_IMAGES_PATH) -> None:
    """
    :param images_data: A dictionary of lists of images to be created
    :param image_save_path: The path to save the images, defaults to IMAGES_PATH (optional)
    :return:
    """

    # images = []
    # decisions = []
    # in_memory_mode = image_save_path != TRAIN_IMAGES_PATH
    # get all png files names in image_save_path subdirectories
    image_save_path.joinpath('LONG').mkdir(exist_ok=True)
    image_save_path.joinpath('SHORT').mkdir(exist_ok=True)
    pngs = [p.name for p in Path(image_save_path).glob(f"*/*.png")]
    for decision, data in images_data.items():
        for image_data in data:
            save_name = str(Path(image_data[0].replace('-', '_')).with_suffix('.png'))
            if save_name in pngs:
                continue
            first = image_data[1]
            to_plot = [create_gaf(x)['gadf'] for x in first]
            create_images(
                x_plots=to_plot,
                image_name=save_name,
                destination=decision,
                folder=image_save_path,
                # save_fig=not in_memory_mode,
            )
    #         images.append(image)
    #         decisions.append(1 if decision == 'LONG' else 0)
    #
    # if in_memory_mode:
    #     return images, decisions


# def generate_gaf_pooled(images_data: dict[str, list], image_save_path=TRAIN_IMAGES_PATH):
#     """
#     A multithreaded version of generate_gaf
#     """
#
#     # in_memory_mode = image_save_path != TRAIN_IMAGES_PATH
#     pngs = [p.name for p in Path(image_save_path).glob(f"*/*.png")]
#
#     def func(data, decision):
#         save_name = str(Path(data[0].replace('-', '_')).with_suffix('.png'))
#         if save_name in pngs:
#             return
#         t1 = time.time()
#         first = data[1]
#         to_plot = [create_gaf(x)['gadf'] for x in first]
#         create_images(
#             x_plots=to_plot,
#             image_name=save_name.replace('.png', ''),
#             destination=decision,
#             folder=image_save_path,
#         )
#         print('generate_gaf_pooled() -> func() Elapsed time:', timedelta(seconds=time.time() - t1))
#
#     # images = []
#     # decisions = []
#     with ThreadPoolExecutor() as executor:
#         for decision, data in images_data.items():
#             futures = [executor.submit(func, data_point, decision) for data_point in data]
#             for future in as_completed(futures):
#                 future.result()
#                 # images.append(image)
#                 # decisions.append(1 if decision == 'LONG' else 0)
#     #
#     # if in_memory_mode:
#     #     return images, decisions


def series_quadrant_to_pil(data: list[pd.Series]) -> Image:
    """
    It takes 4 list of close price series and creates a GADF image from each of them

    :param data: A list of 4 close price series, each at different timeframes
    :return: A PIL image object
    """
    to_plot = [create_gaf(x)['gadf'] for x in data]
    return get_image_from_gaf(x_plots=to_plot)


def trading_action(future_value: float, current_value: float):
    if future_value > current_value:
        return 'LONG'
    else:
        return 'SHORT'


def transform(images: list[Image], image_size: int = 40):
    """
    Given a list of images, resize them to `image_size`, convert them to tensors, and then reshape them to be of
    size 1x40x40x3

    :param images: A list a GAF images
    :param image_size: What to resize the values to

    :return: The image is being transformed into a tensor and then reshaped into a 1x40x40x3 tensor.
    """
    # ToTensor automatically normalizes data to [0 1]
    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    t_images = []
    for image in images:
        image = image.resize((image_size, image_size))
        array = img_to_array(image)
        array /= 255.0
        array.shape = (image_size, image_size, 3)
        t_images.append(array)
    return np.asarray(t_images)


if __name__ == '__main__':
    tmp = Path('/tmp/test_btc_2022')
    tr = '20210101-'
    # threaded = False
    # t1 = time.time()
    # data_to_image_preprocess(timerange=tr)
    # print('Non MultiThreaded Elapsed time:', timedelta(seconds=time.time() - t1))
    threaded = False
    t2 = time.time()
    data_to_image_preprocess(timerange=tr)
    print('MultiThreaded Elapsed time:', timedelta(seconds=time.time() - t2))
    # preprocess = data_to_image_preprocess(
    #     # timerange='20170101-20211231',
    #     data=load_pair_data('BTC/USDT', '30m', timerange='20220223-'),
    #     image_save_path=tmp,
    # )
    # x = preprocess[0]
    # y = preprocess[1]
    # images = quick_gaf(load_pair_data('BTC/USDT', '30m', timerange='20220223-'))
    # images = transform(images)
    # train_dataset = datasets.ImageFolder(
    #     root=str(tmp),
    #     transform=transforms.Compose(
    #         [
    #             transforms.Resize(255),
    #             transforms.ToTensor()
    #             # transforms.Scale(255),
    #         ]
    #     ),
    # )
    # generator = ImageDataGenerator(rescale=1 / 255)
    # data = generator.flow(x=images, batch_size=1)
    # print(torch_x)
    # print(train_dataset[5][0])
    # generate_gaf(preprocess)
