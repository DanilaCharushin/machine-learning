import datetime

import lab6.task3 as task3
import matplotlib.pyplot as plt
from keras.callbacks import Callback


class SaveModelOnConcreteEpochCallback(Callback):
    MODEL_NAME_PREFIX = "my_model"
    EPOCHS = (1, 3, 4, 5)

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 in self.EPOCHS:
            self.model.save(self.get_model_name(epoch))

    def get_model_name(self, epoch: int) -> str:
        return f"{datetime.datetime.now().date()}_{self.MODEL_NAME_PREFIX}_{epoch + 1}.h5"


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = task3.gen_data()

    model = task3.get_model()

    history = model.fit(
        x_train,
        y_train,
        batch_size=32,
        epochs=task3.EPOCHS,
        validation_data=(x_test, y_test),
        callbacks=[SaveModelOnConcreteEpochCallback()],
    )

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
