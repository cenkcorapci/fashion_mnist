import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout

from dl.image_classification_model import ImageClassificationModel


class MobileNET(ImageClassificationModel):

    def __init__(self, optimizer):
        super().__init__(optimizer)
        self._width = 56
        self._height = 56
        self._scale = True

        base_model = MobileNetV2(input_shape=(self._height, self._width, 3),
                                 weights=None,
                                 include_top=False,
                                 pooling='avg')
        output = Dropout(0.5)(base_model.output)
        predict = Dense(10, activation='softmax')(output)

        self._model = Model(inputs=base_model.input, outputs=predict)

        # Specify the training configuration (optimizer, loss, metrics)
        self._model.compile(optimizer=self._optimizer,  # Optimizer
                            # Loss function to minimize
                            loss='categorical_crossentropy',
                            # List of metrics to monitor
                            metrics=['accuracy', 'categorical_crossentropy', 'categorical_accuracy'])

