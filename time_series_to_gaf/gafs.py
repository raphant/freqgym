import time
from datetime import timedelta
from typing import *
from typing import Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
from diskcache import Cache
from lazyft import paths
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import ImageGrid
from numpy import ndarray
from PIL.Image import Image
from pyts.image import GramianAngularField

from time_series_to_gaf.constants import TRAIN_IMAGES_PATH

matplotlib.use('Agg')

cache = Cache(paths.CACHE_DIR)


# Pass times-eries and create a Gramian Angular Field image
# Grab times-eries and draw the charts
@cache.memoize(expire=60 * 60 * 24)
def create_gaf(ts) -> dict[str, Any]:
    """
    :param ts:
    :return:
    """
    t1 = time.time()
    data = dict()
    gadf = GramianAngularField(method='difference', image_size=ts.shape[0])
    data['gadf'] = gadf.fit_transform(pd.DataFrame(ts).T)[0]
    print('GAF elapsed time:', timedelta(seconds=time.time() - t1))
    return data


# Create images of the bundle that we pass
def create_images(
    x_plots: Any,
    image_name: str,
    destination: str,
    image_matrix: tuple = (2, 2),
    folder=TRAIN_IMAGES_PATH,
) -> Image:
    """
    Create a grid of images and save them to disk

    :param x_plots: The list of images to be plotted
    :param image_name: The name of the image
    :param destination: The name of the folder where the images will be saved
    :param image_matrix: tuple = (2, 2)
    :param folder: The folder where the images will be saved
    :return: The path to the image.
    """
    t1 = time.perf_counter()
    try:
        t2 = time.perf_counter()
        fig: Figure = plt.figure(figsize=[img * 4 for img in image_matrix])
        grid = ImageGrid(
            fig,
            111,
            axes_pad=0,
            nrows_ncols=image_matrix,
            share_all=True,
        )
        print('Elapsed time -> create fig & grid:', timedelta(seconds=time.perf_counter() - t2))
        images = x_plots
        t2 = time.perf_counter()
        for image, ax in zip(images, grid):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(image, cmap='rainbow', origin='lower')
        print(
            'create_images() -> zip(images, grid) Elapsed time:',
            timedelta(seconds=time.perf_counter() - t2),
        )
        t2 = time.perf_counter()
        t6 = time.perf_counter()
        save_path = folder / destination / image_name
        print(
            'create_images() -> save_path Elapsed time:',
            timedelta(seconds=time.perf_counter() - t6),
        )
        # save_path.parent.mkdir(exist_ok=True, parents=True)
        # t3 = time.perf_counter()
        # exists = save_path.exists()
        # print(
        #     f'Elapsed time -> exists [{exists}]:', timedelta(seconds=time.perf_counter() - t3)
        # )
        fig.savefig(save_path)
        t4 = time.perf_counter()
        plt.close(fig)
        print('Elapsed time -> plt.close:', timedelta(seconds=time.perf_counter() - t4))
        print(
            'create_images() -> if save_fig Elapsed time:',
            timedelta(seconds=time.perf_counter() - t2),
        )
    except Exception as e:
        raise
    finally:
        print('create_images() Elapsed time:', timedelta(seconds=time.perf_counter() - t1))


@cache.memoize(tag='gaf')
def get_image_from_gaf(
    x_plots,
    image_matrix=(2, 2),
) -> Image:
    """
    It takes a GAF and returns a PIL image

    :param x_plots: a list of images to display
    :param image_matrix: the number of images to display per row and column
    :return: The image of the plot.
    """
    fig: Figure = plt.figure(figsize=[img * 4 for img in image_matrix])
    grid = ImageGrid(
        fig,
        111,
        axes_pad=0,
        nrows_ncols=image_matrix,
        share_all=True,
    )
    images = x_plots
    for image, ax in zip(images, grid):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(image, cmap='rainbow', origin='lower')
    fig.canvas.draw()
    to_image = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close(fig)
    return to_image
