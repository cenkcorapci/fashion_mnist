from abc import ABC, abstractmethod


class ImageClassificationModel(ABC):
    def __init__(self, optimizer):
        self._optimizer = optimizer

    @abstractmethod
    def train(self, X_train, y_train, X_val, y_val, batch_size, epochs, callbacks):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, X_test, y_test):
        raise NotImplementedError
