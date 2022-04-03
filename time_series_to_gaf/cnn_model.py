import datetime as dt
import os
from pathlib import Path

import keras
import pandas as pd
import tensorflow as tf
from keras.initializers import initializers_v2 as initializer
from lazyft.data_loader import load_pair_data
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#  Ensemble CNN network to train a CNN model on GAF images labeled Long and Short
from torchvision import transforms

from time_series_to_gaf.constants import REPO, TEST_IMAGES_PATH, TRAIN_IMAGES_PATH
from time_series_to_gaf.image_loader import ensemble_data
from time_series_to_gaf.preprocess import data_to_image_preprocess, quick_gaf, transform

SPLIT = 0.30
LR = 0.001
TIMESTAMP = dt.datetime.now().strftime("%Y%m%d%H%M%S")


def predict(data: pd.DataFrame = None, model: keras.Sequential = None, pair='BTC/USDT') -> int:
    tmp = Path('/tmp/test_btc_2022')
    if data is None:
        data = load_pair_data(pair, '1h', timerange='20220312-')
    images = quick_gaf(data)
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
    if not model:
        model = create_cnn(40)
        model_to_load = REPO / 'models' / pair.replace('/', '_') / '20220403011222_GlorotUniform.h5'
        try:
            model.load_weights(model_to_load)
        except OSError as e:
            raise OSError(f'Could not load model {model_to_load}') from e
    # x = np.resize(preprocessed[0], 255)

    x = transform(images)
    prediction = model.predict(x[-1])
    print(prediction, type(prediction))
    return prediction[0][0]


def create_cnn(image_size: int, kernel_initializer=None) -> keras.Sequential:
    return keras.Sequential(
        [
            #  First Convolution
            Conv2D(
                32,
                kernel_size=3,
                activation='relu',
                input_shape=(image_size, image_size, 3),
                padding='same',
                kernel_initializer=kernel_initializer,
            ),
            Conv2D(32, kernel_size=3, activation='relu', padding='same'),
            MaxPooling2D(pool_size=2, strides=2),
            Dropout(0.25),
            # Second Convolution
            Conv2D(64, kernel_size=3, activation='relu', padding='same'),
            Conv2D(64, kernel_size=3, activation='relu', padding='same'),
            MaxPooling2D(pool_size=2, strides=2),
            Dropout(0.25),
            # Third Convolution
            Conv2D(128, kernel_size=3, activation='relu', padding='same'),
            Conv2D(128, kernel_size=3, activation='relu', padding='same'),
            MaxPooling2D(pool_size=2, strides=2),
            # Output layer
            Flatten(),
            Dense(1024, activation='relu'),
            Dense(1, activation='sigmoid'),
        ]
    )


def train_and_evaluate(pair: str = 'BTC/USDT'):
    # cnn_networks = 1
    models: list[keras.Sequential] = []
    image_train_path = TRAIN_IMAGES_PATH / pair.replace('/', '_')
    image_test_path = TEST_IMAGES_PATH / pair.replace('/', '_')
    image_train_path.mkdir(parents=True, exist_ok=True)
    image_test_path.mkdir(parents=True, exist_ok=True)
    data_to_image_preprocess(
        timerange='20170101-20211231', image_save_path=image_train_path, pair=pair, interval='1h'
    )
    data_to_image_preprocess(timerange='20220101-', pair=pair, image_save_path=image_test_path)

    # global history, string_list, summary
    target_size = 40
    batch_size = 32
    EPOCHS = 2

    initializers_ = [
        initializer.Orthogonal(),
        initializer.LecunUniform(),
        initializer.VarianceScaling(),
        initializer.RandomNormal(),
        initializer.RandomUniform(),
        initializer.TruncatedNormal(),
        initializer.GlorotNormal(),
        initializer.GlorotUniform(),
        initializer.HeNormal(),
        initializer.HeUniform(),
        initializer.Orthogonal(seed=42),
        initializer.LecunUniform(seed=42),
        initializer.VarianceScaling(seed=42),
        initializer.RandomNormal(seed=42),
        initializer.RandomUniform(seed=42),
        initializer.TruncatedNormal(seed=42),
        initializer.GlorotNormal(seed=42),
        initializer.GlorotUniform(seed=42),
        initializer.HeNormal(seed=42),
        initializer.HeUniform(seed=42),
    ]
    for i, initializer_ in enumerate(initializers_):
        models.append(create_cnn(target_size, kernel_initializer=initializer_))
        # Compile each model
        models[i].compile(
            optimizer=Adam(learning_rate=LR), loss='binary_crossentropy', metrics=['acc']
        )
    # All images will be rescaled by 1./255
    train_validate_datagen = ImageDataGenerator(
        rescale=1 / 255, validation_split=SPLIT
    )  # set validation split
    test_datagen = ImageDataGenerator(rescale=1 / 255)
    # data_chunks = ensemble_data(len(models), str(image_path))
    for j, model in enumerate(models):
        print(f'Net : {initializers_[j].__class__.__name__}')
        train_generator = train_validate_datagen.flow_from_directory(
            directory=image_train_path,
            target_size=(target_size, target_size),
            batch_size=batch_size,
            class_mode='binary',
            subset='training',
        )

        validation_generator = train_validate_datagen.flow_from_directory(
            directory=image_train_path,
            target_size=(target_size, target_size),
            batch_size=batch_size,
            class_mode='binary',
            subset='validation',
        )

        test_generator = test_datagen.flow_from_directory(
            directory=image_test_path,
            target_size=(target_size, target_size),
            batch_size=batch_size,
            class_mode='binary',
        )
        steps_per_epoch = train_generator.samples // train_generator.batch_size
        validation_steps = validation_generator.samples // validation_generator.batch_size
        learning_rate_reduction = ReduceLROnPlateau(
            monitor='val_acc', patience=3, verbose=0, factor=0.5, min_lr=0.00001
        )
        try:
            history = model.fit(
                train_generator,
                epochs=EPOCHS,
                steps_per_epoch=steps_per_epoch,
                validation_data=validation_generator,
                validation_steps=validation_steps,
                callbacks=[learning_rate_reduction],
                verbose=0,
            )
        except Exception as e:
            print(model.summary())
            raise
        print(
            'CNN Model {0:d}: '
            'Epochs={1:d}, '
            'Training Accuracy={2:.5f}, '
            'Validation Accuracy={3:.5f}'.format(
                j + 1, EPOCHS, max(history.history['acc']), max(history.history['val_acc'])
            )
        )

        scores = model.evaluate(test_generator, steps=5, batch_size=batch_size)
        print("Test {0}s: {1:.2f}%".format(models[j].metrics_names[1], scores[1] * 100))
        string_list = []
        model.summary(print_fn=lambda x: string_list.append(x))
        summary = "\n".join(string_list)
        logging = ['{0}: {1}'.format(key, val[-1]) for key, val in history.history.items()]
        log = 'Results:\n' + '\n'.join(logging)
        model_save_path = (
            REPO
            / pair.replace("/", "_")
            / str(TIMESTAMP)
            / 'models'
            / f'{initializers_[j].__class__.__name__}.h5'
        )
        summary_save_path = (
            REPO
            / pair.replace("/", "_")
            / str(TIMESTAMP)
            / 'summaries'
            / f'{initializers_[j].__class__.__name__}.txt'
        )

        model_save_path.parent.mkdir(exist_ok=True, parents=True)
        summary_save_path.parent.mkdir(exist_ok=True, parents=True)

        model.save(model_save_path)
        summary_save_path.write_text(
            f"EPOCHS: {EPOCHS}\nSteps per epoch: {steps_per_epoch}\n"
            f"Validation steps: {validation_steps}\n"
            f"Val Split:{SPLIT}\nLearning RT:{summary}\n\n\n{LR}"
            f"\n\n=========TRAINING LOG========\n{log}"
        )


if __name__ == '__main__':
    train_and_evaluate('BTC/USDT')
    # print(predict())
