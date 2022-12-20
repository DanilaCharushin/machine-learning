from itertools import product

import keras
import numpy as np
from keras import layers, models, optimizers


def logical_operation(a: bool, b: bool, c: bool) -> bool:
    """Вариант 2: (a or b) xor not(b and c)."""

    return (a or b) ^ (not (b and c))


def main():
    # создаем датасет
    data_a = (1, 1)
    data_b = (0, 1)
    data_c = (1, 0)

    data = []
    answers = []
    for a, b, c in product(data_a, data_b, data_c):
        data.append((a, b, c))
        answers.append(logical_operation(a, b, c))

    data = np.asarray(data)
    answers = np.asarray(answers)

    print(f"{data=}")
    print(f"{answers=}")

    # создание слоев
    model_layers = (
        layers.Dense(8, activation="relu", input_shape=(3,), use_bias=False),
        layers.Dense(1, activation="sigmoid", use_bias=False),
    )

    # создаем последовательной модели
    model = models.Sequential(model_layers)
    model.compile(
        optimizer=optimizers.Adam(lr=0.1),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # прогоняем датасет через необученную модель
    naive_before_fit_prediction, np_before_fit_prediction, keras_before_fit_prediction = get_answers(data, model)

    # обучаем модель
    model.fit(
        data,
        answers,
        epochs=50,
        batch_size=2,
    )

    # прогоняем датасет через обученную модель
    naive_after_fit_prediction, np_after_fit_prediction, keras_after_fit_prediction = get_answers(data, model)

    naive_before_fit_prediction = np.asarray(naive_before_fit_prediction)
    np_before_fit_prediction = np.asarray(np_before_fit_prediction)
    naive_after_fit_prediction = np.asarray(naive_after_fit_prediction)
    np_after_fit_prediction = np.asarray(np_after_fit_prediction)

    print(f"CORRECT ANSWERS:\n{answers}")
    print("=====================================================================================")
    print("BEFORE FIT:")
    print(f"\tNAIVE\t\n{naive_before_fit_prediction}")
    print(f"\tNP\t\n{np_before_fit_prediction}")
    print(f"\tKERAS\t\n{keras_before_fit_prediction}")
    print("=====================================================================================")
    print("AFTER FIT:")
    print(f"\tNAIVE\t\n{naive_after_fit_prediction}")
    print(f"\tNP\t\n{np_after_fit_prediction}")
    print(f"\tKERAS\t\n{keras_after_fit_prediction}")


def get_answers(data: np.array, model: keras.Model):
    weights = get_weights(model.layers)

    naive_prediction = []
    np_prediction = []
    for _data in data:
        naive_prediction.append(bool(naive_predict(_data, weights)[0]))
        np_prediction.append(bool(np_predict(_data, weights)[0]))

    keras_prediction = model.predict(data)
    keras_prediction = np.array([bool(round(x[0])) for x in keras_prediction])

    return naive_prediction, np_prediction, keras_prediction


def get_weights(model_layers: tuple) -> list:
    return [layer2 for layer1 in model_layers for layer2 in layer1.weights]


def naive_predict(data, weights):
    _data = data.copy()
    for l_weights in weights:
        _data = naive_process_layer(_data, l_weights)
    return _data


def np_predict(data, weights):
    _data = data.copy()
    for l_weights in weights:
        _data = np.dot(_data, l_weights)
        _data = np.maximum(_data, 0)
    return _data


def naive_process_layer(data, weights):
    assert len(weights.shape) == 2
    assert len(data.shape) == 1
    assert weights.shape[0] == data.shape[0]
    _data = data.copy()
    _output_data = naive_matrix_vector_dot(weights, data)
    _output_data = naive_relu(_output_data)
    return _output_data


def naive_matrix_vector_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    z = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        for j in range(x.shape[0]):
            z[i] += x[j, i] * y[j]
    return z


def naive_relu(x):
    x = x.copy()
    for i in range(x.shape[0]):
        x[i] = max(x[i], 0)
    return x


if __name__ == "__main__":
    main()
