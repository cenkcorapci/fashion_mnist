from abc import ABC, abstractmethod


class ImageClassificationModel(ABC):
    def __init__(self, data_loader, batch_size, input_size, epochs):
        self._data_loader = data_loader
        self._batch_size = batch_size
        self._input_size = input_size
        self._epochs = epochs

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError
