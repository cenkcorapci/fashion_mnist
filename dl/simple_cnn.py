from dl.image_classification_model import ImageClassificationModel


class SimpleCNN(ImageClassificationModel):
    def __init__(self, data_loader, batch_size, input_size, epochs):
        super().__init__(data_loader, batch_size, input_size, epochs)

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError
