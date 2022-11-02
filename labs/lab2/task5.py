import warnings

import matplotlib.cbook
import matplotlib.colors as mclr
import matplotlib.pyplot as plt
import numpy as np
from keras import layers, models

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

SIZE = 1000
VALIDATION_PERCENT = 0.1
EPOCHS = 20
BATCH_SIZE = 64


def genData(size=SIZE):
    size1 = size // 2
    size2 = size - size1
    x1 = np.random.rand(size1, 1) * 1.3 - 0.95
    y1 = np.asarray([3.5 * (i + 0.2) ** 2 - 0.8 + (np.random.rand(1) - 0.5) / 3 for i in x1])
    data1 = np.hstack((x1, y1))
    label1 = np.zeros([size1, 1])
    div1 = round(size1 * 0.8)
    x2 = np.random.rand(size2, 1) * 1.3 - 0.35
    y2 = np.asarray([-3.5 * (i - 0.2) ** 2 + 0.8 + (np.random.rand(1) - 0.5) / 3 for i in x2])
    data2 = np.hstack((x2, y2))
    label2 = np.ones([size2, 1])
    div2 = round(size2 * 0.8)
    div = div1 + div2
    order = np.random.permutation(div)
    train_data = np.vstack((data1[:div1], data2[:div2]))
    test_data = np.vstack((data1[div1:], data2[div2:]))
    train_label = np.vstack((label1[:div1], label2[:div2]))
    test_label = np.vstack((label1[div1:], label2[div2:]))

    return (train_data[order, :], train_label[order, :]), (test_data, test_label)


# Функцию выбрать в зависимости от варианта
def drawResults(data, label, prediction):
    p_label = np.array([round(x[0]) for x in prediction])
    plt.scatter(
        data[:, 0],
        data[:, 1],
        s=30,
        c=label[:, 0],
        cmap=mclr.ListedColormap(["red", "blue"]),
    )
    plt.scatter(
        data[:, 0],
        data[:, 1],
        s=10,
        c=p_label,
        cmap=mclr.ListedColormap(["red", "blue"]),
    )
    plt.grid()
    plt.show()


(train_data, train_label), (test_data, test_label) = genData()

##################

# создание последовательной модели
model = models.Sequential()
# добавление слоев
model.add(layers.Dense(16, activation="relu", input_shape=(2,)))
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", metrics=["accuracy"])

validation_size = int(SIZE * min(VALIDATION_PERCENT, 0.5))

x_val = train_data[:validation_size]
partial_x_train = train_data[validation_size:]
y_val = train_label[:validation_size]
partial_y_train = train_label[validation_size:]

history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_val, y_val),
)

##################

# Получение ошибки и точности в процессе обучения
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

# Получение и вывод результатов на тестовом наборе
results = model.evaluate(test_data, test_label)
print(results)

# Вывод результатов бинарной классификации
all_data = np.vstack((train_data, test_data))
all_label = np.vstack((train_label, test_label))
pred = model.predict(all_data)
drawResults(all_data, all_label, pred)
