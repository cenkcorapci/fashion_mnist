import tensorflow as tf

from dl.image_classification_model import ImageClassificationModel
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

class VGGModelType:
    vgg16 = 'vgg16'
    vgg19 = 'vgg19'


class VGG(ImageClassificationModel):
    def __init__(self, optimizer, model_type: VGGModelType = VGGModelType.vgg16):
        super().__init__(optimizer)
        if model_type == VGGModelType.vgg16:
            self._model = tf.keras.applications.resnet_v2.ResNet50V2(input_shape=(28, 28, 1), weights=None, include_top=False)
        else:
            self._model = tf.keras.applications.resnet_v2.ResNet50V2(input_shape=(28, 28, 1), weights=None, include_top=False)

        new_layer = Dense(10, activation='softmax', name='classification')

        inp = self._model.input
        out = new_layer(self._model.layers[-1].output)

        self._model = Model(inp, out)
        self._model.summary(line_length=150)
        # Specify the training configuration (optimizer, loss, metrics)
        self._model.compile(optimizer=self._optimizer,  # Optimizer
                            # Loss function to minimize
                            loss='categorical_crossentropy',
                            # List of metrics to monitor
                            metrics=['accuracy', 'categorical_crossentropy', 'categorical_accuracy'])

    def train(self, data_generator, X_train, y_train, X_val, y_val, batch_size, epochs, callbacks):
        data_generator.fit(X_train)
        # fits the model on batches with real-time data augmentation:

        history = self._model.fit_generator(data_generator.flow(X_train, y_train, batch_size=batch_size),
                                            steps_per_epoch=len(X_train) / batch_size, epochs=epochs,
                                            verbose=1,
                                            callbacks=callbacks,
                                            validation_data=(X_val, y_val))
        return history


if __name__ == '__main__':
    m = VGG(tf.keras.optimizers.SGD())
