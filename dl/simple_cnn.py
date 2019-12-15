from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential

from dl.image_classification_model import ImageClassificationModel


class SimpleCNN(ImageClassificationModel):
    def __init__(self, optimizer):
        super().__init__(optimizer)

        # input image dimensions

        self._model = Sequential()
        self._model.add(Conv2D(32, kernel_size=(3, 3),
                               activation='relu',
                               kernel_initializer='he_normal',
                               input_shape=(28, 28, 1)))
        self._model.add(MaxPooling2D((2, 2)))
        self._model.add(Dropout(0.25))
        self._model.add(Conv2D(64, (3, 3), activation='relu'))
        self._model.add(MaxPooling2D(pool_size=(2, 2)))
        self._model.add(Dropout(0.25))
        self._model.add(Conv2D(128, (3, 3), activation='relu'))
        self._model.add(Dropout(0.4))
        self._model.add(Flatten())
        self._model.add(Dense(128, activation='relu'))
        self._model.add(Dropout(0.3))
        self._model.add(Dense(10, activation='softmax'))

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

    def evaluate(self, X_test, y_test):
        score = self._model.evaluate(X_test, y_test, verbose=0)
        return score
