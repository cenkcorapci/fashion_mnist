from IPython.display import clear_output
from matplotlib import pyplot as plt
from tensorflow import keras


class PlotLosses(keras.callbacks.Callback):
    def __init__(self):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('accuracy'))
        self.val_losses.append(logs.get('val_accuracy'))
        self.i += 1

        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="accuracy")
        plt.plot(self.x, self.val_losses, label="val_accuracy")
        plt.legend()
        plt.show()
