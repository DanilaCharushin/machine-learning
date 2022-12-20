import random

import matplotlib.pyplot as plt
import numpy as np
from keras import layers, models
from keras.utils import to_categorical
from lab6 import gens

IMAGE_SIZE = 50
DATASET_SIZE = 500
CHECK_SIZE = 100
VALIDATION_PERCENT = 0.2
EPOCHS = 10


def gen_data(size=DATASET_SIZE, img_size=IMAGE_SIZE):
    c1 = size // 2
    c2 = size - c1

    val_c = int(size * VALIDATION_PERCENT // 2)

    # 1 - horizontal, 0 - vertical

    label_c1 = np.full([c1, 1], 1)
    data_c1 = np.array([gens.gen_h_line(img_size) for _ in range(c1)])

    data_c1_train = data_c1[val_c:]
    data_c1_test = data_c1[:val_c]

    label_c1_train = label_c1[val_c:]
    label_c1_test = label_c1[:val_c]

    label_c2 = np.full([c2, 1], 0)
    data_c2 = np.array([gens.gen_v_line(img_size) for _ in range(c2)])

    data_c2_train = data_c2[val_c:]
    data_c2_test = data_c2[:val_c]

    label_c2_train = label_c2[val_c:]
    label_c2_test = label_c2[:val_c]

    data_train = np.vstack((data_c1_train, data_c2_train))
    label_train = np.vstack((label_c1_train, label_c2_train))

    data_test = np.vstack((data_c1_test, data_c2_test))
    label_test = np.vstack((label_c1_test, label_c2_test))

    data_train = np.expand_dims(data_train, axis=3)
    data_test = np.expand_dims(data_test, axis=3)

    label_train = to_categorical(label_train)
    label_test = to_categorical(label_test)

    return data_train, label_train, data_test, label_test


def get_model() -> models.Model:
    """Создание и компиляция последовательной модели."""

    model = models.Sequential(
        [
            layers.Conv2D(
                filters=32,
                kernel_size=3,
                padding="same",
                activation="relu",
                input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
                strides=1,
            ),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            layers.Conv2D(
                filters=16,
                kernel_size=3,
                activation="relu",
            ),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            layers.Flatten(),
            layers.Dense(
                units=2,
                activation="softmax",
            ),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = gen_data()

    model = get_model()

    print(model.summary())

    history = model.fit(
        x_train,
        y_train,
        batch_size=32,
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
    )
    validation_score = model.evaluate(x_test, y_test)
    print(validation_score)

    x_val, y_val, _, _ = gen_data(CHECK_SIZE)

    val_half = int(CHECK_SIZE * (1 - VALIDATION_PERCENT)) // 2

    limits = (
        (0, val_half),
        (val_half, 2 * val_half - 1),
    )
    for limits_ in limits:
        for _ in range(5):
            index = random.randint(limits_[0], limits_[1])
            plt.imshow(x_val[index])
            plt.colorbar()
            plt.show()
            print(y_val[index])

    x_val = np.expand_dims(x_val, axis=3)
    y_val = to_categorical(y_val)

    validation_score = model.evaluate(x_val, y_val)
    print(validation_score)

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    epochs = range(1, len(loss) + 1)

    # Построение графика ошибки
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Построение графика точности
    plt.clf()
    plt.plot(epochs, acc, "bo", label="Training acc")
    plt.plot(epochs, val_acc, "b", label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
