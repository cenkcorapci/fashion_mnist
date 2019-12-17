import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import Dense, Dropout
from tqdm import tqdm

from dl.image_classification_model import ImageClassificationModel


class MobileNET(ImageClassificationModel):
    _height = 56
    _width = 56

    def __init__(self, optimizer):
        super().__init__(optimizer)

        base_model = MobileNet(input_shape=(MobileNET._height, MobileNET._width, 3),
                               include_top=False,
                               pooling='avg')
        output = Dropout(0.5)(base_model.output)
        predict = Dense(10, activation='softmax')(output)

        self._model = Model(inputs=base_model.input, outputs=predict)

        # Specify the training configuration (optimizer, loss, metrics)
        self._model.compile(optimizer=self._optimizer,  # Optimizer
                            # Loss function to minimize
                            loss='sparse_categorical_crossentropy',
                            # List of metrics to monitor
                            metrics=['accuracy', 'categorical_crossentropy', 'categorical_accuracy'])

    def train(self, data_generator, X_train, y_train, X_val, y_val, batch_size, epochs, callbacks):
        data_generator.fit(X_train)
        # fits the model on batches with real-time data augmentation:
        print(X_train.shape)

        X_t = X_train.reshape((-1, 28, 28))
        X_t = np.array(
            [self._scale_image(x) for x in tqdm(iter(X_t), desc='Resizing training images')])
        X_v = X_val.reshape((-1, 28, 28))
        X_v = np.array(
            [self._scale_image(x) for x in tqdm(iter(X_v), desc='Resizing training images')])
        print(X_t.shape)
        history = self._model.fit_generator(data_generator.flow(X_t, y_train, batch_size=batch_size),
                                            steps_per_epoch=len(X_t) / batch_size, epochs=epochs,
                                            verbose=1,
                                            callbacks=callbacks,
                                            validation_data=(X_v, y_val))
        return history

    def _scale_image(self, x):
        img = np.array(Image.fromarray(x).resize((MobileNET._height, MobileNET._width)))
        img = np.expand_dims(img, axis=3)
        return np.repeat(img, 3, axis=2).astype(float)


if __name__ == '__main__':
    m = MobileNET(tf.keras.optimizers.Adam())
