from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import classification_report, confusion_matrix


class ImageClassificationModel(ABC):
    def __init__(self, optimizer):
        self._optimizer = optimizer
        self._model = None

    @abstractmethod
    def train(self, data_generator, X_train, y_train, X_val, y_val, batch_size, epochs, callbacks):
        raise NotImplementedError

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
