import efficientnet.tfkeras as efn

from dl.image_classification_model import ImageClassificationModel


class EfficientNet(ImageClassificationModel):
    def __init__(self, optimizer):
        super().__init__(optimizer)

        # input image dimensions

        self._model = efn.EfficientNetB0()

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
    from tensorflow.keras.optimizers import SGD
    import tensorflow as tf

    with tf.compat.v1.device('/devices/XLA_GPU:0'):
        conf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=conf))
        m = EfficientNet(SGD())
