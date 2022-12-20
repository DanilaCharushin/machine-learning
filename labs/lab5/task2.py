"""
Есть 3 режима работы программы:
1. Создание датасета:
    MODE = GENERATE_DATA
2. Обучение моделей по датасету и вывод графиков и выходного файла
    MODE = FIT_MODEL_AND_READ_DATA
3. Вывод графиков и выходного файла
    MODE = READ_DATA

"""
import json
import random
from typing import Callable, Dict

import keras
import numpy as np
import pandas as pd
from keras import Input, layers, losses, optimizers
from keras.layers import Dense
from keras.saving.save import load_model
from matplotlib import pyplot as plt


class Mode:
    GENERATE_DATA = 1
    FIT_MODEL_AND_READ_DATA = 2
    READ_DATA = 3


MODE = Mode.FIT_MODEL_AND_READ_DATA

SIZE = 4000  # размер выборки
DIV = 0.85  # 0.8 - тренировочная, 0.2 - тестовая
EPOCHS = 100
ENCODING_DIM = 4

PATH_TO_IN_DATA = "in_data_lab5.csv"  # путь к файлу с данными
PATH_TO_OUT_DATA = "out_data_lab5.csv"  # путь к файлу с данными
PATH_TO_MODEL = "lab5.h5"
PATH_TO_ENCODER = "encoder.h5"
PATH_TO_DECODER = "decoder.h5"
PATH_TO_REGRESSION = "regression.h5"

ATTRIB_FUNCTIONS: Dict[int, Callable[[float, float], float]] = {
    0: lambda x, e: np.log(np.abs(x)) + e,
    1: lambda x, e: np.sin(3 * x) + e,
    2: lambda x, e: np.exp(x) + e,
    3: lambda x, e: x + 4 + e,
    4: lambda x, e: x + np.sqrt(np.abs(x)) + e,
    5: lambda x, e: x + e,
}


def target_function(x: float, e: float) -> float:
    return -(x ** 3) + e


def rand_x():
    return round(random.uniform(-5, 10), 3)


def rand_e():
    return round(random.uniform(0, 0.3), 3)


def generate_data() -> pd.DataFrame:
    attrib = [[] for _ in range(len(ATTRIB_FUNCTIONS))]  # признаки
    target = []  # целевое значение

    X = np.random.uniform(-5, 10, SIZE)
    X.sort()

    for x in X:
        e = rand_e()
        for i in range(len(ATTRIB_FUNCTIONS)):
            attrib[i].append(ATTRIB_FUNCTIONS[i](x, e))
        target.append(target_function(x, e))

    colors = [
        "#f08",
        "#00AA00",
        "#00a2ff",
        "#0400ff",
        "#123333",
        "#FFAA00",
    ]
    labels = [
        "np.log(np.abs(x)) + e",
        "np.sin(3 * x) + e",
        "np.e ^ x + e",
        "x + 4 + e",
        "x + np.sqrt(np.abs(x)) + e",
        "x + e",
    ]
    for attrib_, label, color in zip(attrib, labels, colors):
        plt.plot(X, attrib_, c=color, label=label)
        plt.title("Attribs")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()

    plt.plot(X, target, c="#FF0000", label="-(x^3) + e")
    plt.title("Target")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

    return pd.DataFrame(
        {
            "Признак 1": attrib[0],
            "Признак 2": attrib[1],
            "Признак 3": attrib[2],
            "Признак 4": attrib[3],
            "Признак 5": attrib[4],
            "Признак 6": attrib[5],
            "Цель": target,
        }
    )


def divide_data(data_df):
    _div = round(DIV * 100 // 10)
    train_feature = data_df.iloc[data_df.index % _div != 0, 0:6]  # срезы выборок
    test_feature = data_df.iloc[data_df.index % _div == 0, 0:6]
    train_target = data_df.iloc[data_df.index % _div != 0, 6:7]  # срезы целевых значений (target)
    test_target = data_df.iloc[data_df.index % _div == 0, 6:7]
    return train_feature, test_feature, train_target, test_target


def generate_models():
    main_input = Input(shape=(6,), dtype="float32", name="main_input")
    encoder_output = Dense(ENCODING_DIM, activation="relu", name="encoder_output")(main_input)
    encoded_input = keras.Input(shape=(ENCODING_DIM,))

    decoded = layers.Dense(6, activation="linear", name="decoded_layer")(encoder_output)

    x1_tensor = layers.Dense(64, activation="relu", name="x1")(encoder_output)
    x2_tensor = layers.Dense(128, activation="relu", name="x2")(x1_tensor)
    regression_layer_tensor = layers.Dense(1, activation="linear", name="regression_layer")(x2_tensor)

    autoencoder = keras.Model(main_input, outputs=[decoded, regression_layer_tensor])

    decoder_layer = autoencoder.get_layer("decoded_layer")
    regression_layer = autoencoder.get_layer("regression_layer")
    x1 = autoencoder.get_layer("x1")
    x2 = autoencoder.get_layer("x2")
    opt = keras.optimizers.Adam(learning_rate=0.005)
    autoencoder.compile(
        optimizer=opt,
        loss={
            "decoded_layer": losses.mean_squared_error,
            "regression_layer": losses.mean_squared_error,
        },
    )

    encoder = keras.Model(main_input, encoder_output)
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
    regression = keras.Model(encoded_input, regression_layer(x2(x1(encoded_input))))

    return autoencoder, encoder, decoder, regression


def plot_graphs(_x, _y, labels, start=0, end=None):
    colors = [
        "#f08",
        "#f00",
        "#00a2ff",
        "#0400ff",
    ]
    if end is None:
        end = len(_x) + 1
    for y, label, color in zip(_y, labels, colors):
        plt.plot(_x[start:end], y[start:end], c=color, label=label)
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def main(*, fit_model: bool):
    data_df = pd.read_csv(PATH_TO_IN_DATA, index_col=0)

    train_feature, test_feature, train_target, test_target = divide_data(data_df)

    if fit_model:
        autoencoder, encoder, decoder, regression = generate_models()

        H = autoencoder.fit(
            train_feature,
            {
                "decoded_layer": train_feature,
                "regression_layer": train_target,
            },
            epochs=EPOCHS,
            batch_size=256,
            validation_data=(
                test_feature,
                {
                    "decoded_layer": test_feature,
                    "regression_layer": test_target,
                },
            ),
        )

        decoded_loss = H.history["decoded_layer_loss"]
        val_decoded_loss = H.history["val_decoded_layer_loss"]
        regression_layer_loss = H.history["regression_layer_loss"]
        val_regression_layer_loss = H.history["val_regression_layer_loss"]

        decoder.save(PATH_TO_DECODER)
        encoder.save(PATH_TO_ENCODER)
        regression.save(PATH_TO_REGRESSION)
        autoencoder.save(PATH_TO_MODEL)

        with open("H.json", "w") as f:
            f.write(json.dumps([decoded_loss, val_decoded_loss, regression_layer_loss, val_regression_layer_loss]))

    with open("H.json", "r") as f:
        decoded_loss, val_decoded_loss, regression_layer_loss, val_regression_layer_loss = json.load(f)

    epochs = list(range(len(decoded_loss)))

    labels = [
        "decoded_loss",
        "val_decoded_loss",
        "regression_layer_loss",
        "val_regression_layer_loss",
    ]
    y = [
        decoded_loss,
        val_decoded_loss,
        regression_layer_loss,
        val_regression_layer_loss,
    ]
    plot_graphs(epochs, y, labels)
    plot_graphs(epochs, y, labels, start=30)
    plot_graphs(epochs, y, labels, start=80)

    decoder = load_model(PATH_TO_DECODER)
    encoder = load_model(PATH_TO_ENCODER)
    regression = load_model(PATH_TO_REGRESSION)

    encoded_data = encoder.predict(test_feature)
    decoded_data = decoder.predict(encoded_data)
    regression_data = regression.predict(encoded_data)

    out = test_feature.copy()
    out["Закодировано"] = [item for item in encoded_data]
    out["Декодировано"] = [item for item in decoded_data]
    out["Цель"] = test_target["Цель"].tolist()
    out["Регрессия"] = regression_data

    out.to_csv(PATH_TO_OUT_DATA)


if __name__ == "__main__":
    if MODE == Mode.GENERATE_DATA:
        generated_data = generate_data()
        generated_data.to_csv(PATH_TO_IN_DATA)
    elif MODE == Mode.FIT_MODEL_AND_READ_DATA:
        main(fit_model=True)
    elif MODE == Mode.READ_DATA:
        main(fit_model=False)
