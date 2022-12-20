import math
import random

import matplotlib.pyplot as plt
import numpy as np
from keras import layers, Sequential

SIZE = 1500


def gen_sequence_example(seq_len=SIZE):
    seq = [math.sin(i / 10) + random.normalvariate(0, 0.09) for i in range(seq_len)]
    return np.array(seq)


def gen_sequence(seq_len=SIZE):
    seq = [math.sin(i / 5) * math.sin(i / 10 + 0.5) + random.normalvariate(0, 0.04) for i in range(seq_len)]
    return np.array(seq)


def draw_sequence():
    seq = gen_sequence(SIZE)
    plt.plot(range(len(seq)), seq)
    plt.show()


draw_sequence()


def gen_data_from_sequence(seq_len=SIZE, lookback=10):
    seq = gen_sequence(seq_len)
    past = np.array([[[seq[j]] for j in range(i, i + lookback)] for i in range(len(seq) - lookback)])
    future = np.array([[seq[i]] for i in range(lookback, len(seq))])
    return past, future


data, res = gen_data_from_sequence()

dataset_size = len(data)
train_size = (dataset_size // 10) * 7
val_size = (dataset_size - train_size) // 2

train_data, train_res = data[:train_size], res[:train_size]
val_data, val_res = data[train_size : train_size + val_size], res[train_size : train_size + val_size]
test_data, test_res = data[train_size + val_size :], res[train_size + val_size :]

model = Sequential()
model.add(layers.GRU(32, recurrent_activation="sigmoid", input_shape=(None, 1), return_sequences=True))
model.add(layers.LSTM(32, activation="relu", input_shape=(None, 1), return_sequences=True, dropout=0.2))
model.add(layers.GRU(32, input_shape=(None, 1), recurrent_dropout=0.2))
model.add(layers.Dense(1))

model.compile(optimizer="nadam", loss="mse")
history = model.fit(train_data, train_res, epochs=50, validation_data=(val_data, val_res))

loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)

# Построение графика ошибки
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

predicted_res = model.predict(test_data)
pred_length = range(len(predicted_res))

plt.clf()
plt.plot(pred_length, predicted_res, "b", label="Predicted")
plt.plot(pred_length, test_res, "g", label="Original")
plt.title("Predicted and original signal")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
