import gym
import numpy as np
import torch as th
from keras_preprocessing.image import ImageDataGenerator
from lazyft.data_loader import load_and_populate_pair_data, load_pair_data
from pandas import DataFrame
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from series2gaf import GenerateGAF
from time_series_to_gaf.constants import TRAIN_IMAGES_PATH
from time_series_to_gaf.image_loader import ensemble_data


class CustomCNN2(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 3):
        super(CustomCNN2, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]

        # The GAF-CNN model works well with the simple neural architecture, two convolutional layers with 16 kernels, and
        # one fully-connected layer with 128 denses. The max-pooling layer, which uses general picture classification, calculates
        # the maximum value for each patch of the feature map usually.
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=(5, 5), padding="same"),
            nn.ReLU(),
        )
        self.max_pool = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(5, 5), padding="same"),
            nn.ReLU(),
        )
        self.fc = nn.Linear(16 * 5 * 5, 128)
        # FC 9 + softmax
        self.fc9 = nn.Sequential(
            nn.Linear(128, features_dim),
            nn.Softmax(dim=3),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Given a batch of images, the function first applies a convolutional layer with a relu activation
        function,
        then a max pooling layer, then another convolutional layer with a relu activation function,
        then a max pooling layer, then a dropout layer, then a flatten layer, then a fully connected
        layer with a relu activation function,
        then a dropout layer, and finally a fully connected layer

        :param observations: The input to the neural network
        :type observations: th.Tensor
        :return: The output of the last linear layer.
        """
        x = self.conv1(observations)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.fc(x)
        x = self.fc9(x)
        return x


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 3):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]

        self.conv1 = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=(5, 5), padding="same"),
            nn.ReLU(),
        )
        # The max pooling layer makes the image unclear for the human eye by sampling it down to a lower resolution,
        # but for the machine learning model it mostly removes not relevant elements and makes it more robust to changes
        # in the input (like rotation, shifting, translation etc.)
        self.max_pool = nn.MaxPool2d((2, 2))
        # one fully-connected layer with 128 dense units
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 128),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(nn.Conv2d(16, 36, kernel_size=(5, 5), padding="same"), nn.ReLU())
        self.dropout1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Sequential(nn.Linear(200, features_dim), nn.ReLU())  # relu
        self.dropout2 = nn.Dropout(0.5)
        self.linear2 = nn.Sequential(nn.Linear(200, 3), nn.Softmax())  # softmax

        self.cnn = nn.Sequential(
            self.conv1,
            self.max_pool,
            self.conv2,
            self.dropout1,
            self.flatten,
            self.linear1,
            self.dropout2,
            self.linear2,
        )

        # Compute shape by doing one forward pass
        # with th.no_grad():
        #     n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Given a batch of images, the function first applies a convolutional layer with a relu activation
        function,
        then a max pooling layer, then another convolutional layer with a relu activation function,
        then a max pooling layer, then a dropout layer, then a flatten layer, then a fully connected
        layer with a relu activation function,
        then a dropout layer, and finally a fully connected layer

        :param observations: The input to the neural network
        :type observations: th.Tensor
        :return: The output of the last linear layer.
        """
        out = self.max_pool(self.conv1(observations))
        out = self.max_pool(self.conv2(out))
        out = self.dropout1(out)
        out = self.flatten(out)
        out = self.linear1(out)
        out = self.dropout2(out)

        return out


class CustomCNN3(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 0):
        super().__init__(observation_space, features_dim)
        # first convolution
        self.conv11 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        # batch normalization
        self.conv12 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding='same',
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.4),
        )
        # second convolution
        self.conv21 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

        self.conv22 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.conv23 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding='same',
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.4),
        )
        # third convolution
        self.conv31 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=128,
                kernel_size=4,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :param observations: (batch_size, 3, height, width)
        :return: (batch_size, features_dim)
        """
        # first convolution
        x = self.conv11(observations)
        x = self.conv12(x)
        x = self.conv13(x)
        # second convolution
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.conv23(x)
        # third convolution
        x = self.conv31(x)
        return x


if __name__ == '__main__':
    # random_series = np.random.uniform(low=50.0, high=150.0, size=(200,))
    # random_series2 = np.random.uniform(low=50.0, high=150.0, size=(200,))
    # random_series3 = np.random.uniform(low=50.0, high=150.0, size=(200,))
    # random_series4 = np.random.uniform(low=50.0, high=150.0, size=(200,))
    # timeSeries = list(random_series)
    # ohlc: DataFrame = load_pair_data('BTC/USDT', '1h', timerange='20210101-20220101')
    # # drop date and volume
    # ohlc.drop(columns=['date', 'volume'], inplace=True)
    # ohlc.dropna(inplace=True)
    # ohlc.reset_index(drop=True, inplace=True)
    # time_series: np.ndarray = ohlc.to_numpy()
    # print('ohlc shape:', time_series.shape)
    # print('ohlc head:', time_series[0])
    # windowSize = 50
    # rollingLength = 10
    # # timeSeries = np.array([random_series, random_series2, random_series3, random_series4]).T
    # fileName = 'demo_%02d_%02d' % (windowSize, rollingLength)
    # GenerateGAF(
    #     all_ts=time_series,
    #     window_size=windowSize,
    #     rolling_length=rollingLength,
    #     fname=fileName,
    # )
    # gaf = np.load('%s_gaf.pkl' % fileName, allow_pickle=True)
    # gaf = np.reshape(gaf, (gaf.shape[0], gaf.shape[1], gaf.shape[2], 1))
    # print('gaf shape:', gaf.shape)
    # # print(gaf[0, 0])
    # # box with shape: (gaf.shape[1], gaf.shape[2], 1)
    # # box = gym.spaces.Box(low=0, high=1, shape=(gaf.shape[1], gaf.shape[2], 1))
    # # print(box.shape, box.sample())
    #
    # # CustomCNN2(box).forward(th.as_tensor(box.sample()))
    # todo: decide how to approach ensemble and change this variable
    n_ensemble = 1
    # train_dataset = datasets.ImageFolder(
    #     root=str(IMAGES_PATH),
    #     transform=transforms.Compose(
    #         [
    #             transforms.Resize(255),
    #             # transforms.Scale(255),
    #             transforms.ToTensor(),
    #         ]
    #     ),
    # )
    data_chunks = ensemble_data(n_ensemble, str(TRAIN_IMAGES_PATH))
    train_validate_datagen = ImageDataGenerator(rescale=1 / 255, validation_split=0.30)
    df_train = data_chunks[0].iloc[:-60]
    df_test = data_chunks[0].iloc[-60:]
    train_generator = train_validate_datagen.flow_from_dataframe(
        dataframe=df_train,
        directory=str(TRAIN_IMAGES_PATH),
        target_size=(225, 225),
        x_col='Images',
        y_col='Labels',
        batch_size=32,
        class_mode='binary',
        subset='training',
    )
    print(train_generator.next())
    print(train_dataset[0])
    exit()
    print(f'Train dataset size: {len(train_dataset)}')
    # print(f'Train dataset shape: {train_dataset[0][0].show()}')
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        # shuffle=True,
    )
    next(iter(train_loader))
    next(iter(train_loader))
    batch = next(iter(train_loader))
    print(batch[0][0])
    obs_space = gym.spaces.Box(
        low=0,
        high=255,
        shape=(3, 255, 255),
        dtype=np.float32,
    )
    print(obs_space.sample())
