from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import *

from dl.image_classification_model import ImageClassificationModel


class SimpleCNN(ImageClassificationModel):
    def __init__(self, optimizer):
        super().__init__(optimizer)

        self._height = 28
        self._width = 28
        self._scale = False

        # input image dimensions

        input = Input(shape=[28, 28, 1])
        x = Conv2D(32, (5, 5), strides=1, padding='same')(input)
        # x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)
        x = Conv2D(32, (5, 5), strides=1, padding='same')(x)
        # x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
        # x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)
        # x = Dropout (0.5)(x)
        x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
        # x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
        # x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)

        x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
        x = Dropout(0.35)(x)
        x = Flatten()(x)
        x = Dense(200)(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dense(10)(x)
        x = Activation('softmax')(x)
        self._model = Model(inputs=input, outputs=x)

        # Specify the training configuration (optimizer, loss, metrics)
        self._model.compile(optimizer=self._optimizer,  # Optimizer
                            # Loss function to minimize
                            loss='categorical_crossentropy',
                            # List of metrics to monitor
                            metrics=['accuracy', 'categorical_crossentropy', 'categorical_accuracy'])


if __name__ == '__main__':
    from tensorflow.keras.optimizers import SGD

    m = SimpleCNN(SGD())
    m.load_from_best_checkpoint('simple_cnn_adam')
