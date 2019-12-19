from abc import ABC
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from pathlib import Path
from config import DL_MODELS_PATH
import logging


class ImageClassificationModel(ABC):
    def __init__(self, optimizer):
        self._optimizer = optimizer
        self._model = None

        self._width = 56
        self._height = 56
        self._scale = True

    def train(self, data_generator, X_train, y_train, X_val, y_val, batch_size, epochs, callbacks):
        data_generator.fit(X_train)
        X_t = X_train
        X_v = X_val
        if self._scale:
            X_t, X_v = self.scale_inputs(X_train, X_val)
        history = self._model.fit_generator(data_generator.flow(X_t, y_train, batch_size=batch_size, shuffle=True),
                                            steps_per_epoch=len(X_t) / batch_size, epochs=epochs,
                                            verbose=1,
                                            callbacks=callbacks,
                                            validation_data=(X_v, y_val))
        return history

    def evaluate(self, X_test, y_test, plot_confussion=False):
        # Confution Matrix and Classification Report
        Y_pred = self._model.predict(X_test)
        y_pred = np.argmax(Y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)
        if plot_confussion:
            print('Confusion Matrix')
            fig = go.Figure(data=go.Heatmap(
                z=confusion_matrix(y_true, y_pred)))
            fig.show()

        target_names = ["top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag",
                        "ankle boot"]
        report = pd.DataFrame(classification_report(y_true, y_pred, target_names=target_names, output_dict=True))
        return report

    def _scale_image(self, x):
        img = np.array(Image.fromarray(x).resize((self._height, self._width)))
        img = np.expand_dims(img, axis=3)
        return np.repeat(img, 3, axis=2).astype(float)

    def load_from_best_checkpoint(self, model_name, checkpoints_path=DL_MODELS_PATH):
        best = 0
        path_of_best_weights = None
        for weight_file in Path(checkpoints_path).rglob('*.hdf5'):
            p = str(weight_file)
            if model_name in p:
                accuracy = int(p.split('/')[-1].split('.')[-2])
                if accuracy >= best:
                    path_of_best_weights = p
        if path_of_best_weights is not None:
            self._model = tf.keras.models.load_model(path_of_best_weights)
            logging.info(f'Loaded {path_of_best_weights}')
        else:
            logging.error('Can not find any model checkpoint')

    def scale_inputs(self, X_train, X_val):
        X_t = X_train.reshape((-1, 28, 28))
        X_t = np.array(
            [self._scale_image(x) for x in tqdm(iter(X_t), desc='Resizing training images')])
        X_v = X_val.reshape((-1, 28, 28))
        X_v = np.array(
            [self._scale_image(x) for x in tqdm(iter(X_v), desc='Resizing training images')])
        return X_t, X_v
