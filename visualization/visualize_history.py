import random

import matplotlib
import plotly.graph_objs as go
from plotly import subplots
from plotly.offline import iplot

from data.metrics import get_tensorboard_scalars

hex_colors_only = []

for name, hex in matplotlib.colors.cnames.items():
    hex_colors_only.append(hex)


def create_trace(x, y, ylabel, color):
    trace = go.Scatter(
        x=x, y=y,
        name=ylabel,
        marker=dict(color=color),
        mode="markers+lines",
        text=x
    )
    return trace


def plot_accuracy_and_loss(histories=None):
    """
    This will plot accuracies and lossesfor both training and validation sessions for given history objects.
    If no history object is given, values will be fetched from Tensorboard logs.
    args:
        histories (Keras history): result of keras training
    """
    fig = subplots.make_subplots(rows=2, cols=2, subplot_titles=('Training accuracy', 'Validation accuracy',
                                                                 'Training loss ', 'Validation loss'))

    def append_trace(model_name, acc, val_acc, loss, val_loss, epochs):
        e = list(range(epochs))
        color = random.choice(hex_colors_only)
        trace_ta = create_trace(e, acc, model_name, color)
        trace_va = create_trace(e, val_acc, model_name, color)
        trace_tl = create_trace(e, loss, model_name, color)
        trace_vl = create_trace(e, val_loss, model_name, color)

        fig.append_trace(trace_ta, 1, 1)
        fig.append_trace(trace_va, 1, 2)
        fig.append_trace(trace_tl, 2, 1)
        fig.append_trace(trace_vl, 2, 2)

    if histories is None:
        df_accuracies, df_losses = get_tensorboard_scalars()
        for model_name in df_accuracies.model_name.unique():
            df_acc = df_accuracies.loc[df_accuracies.model_name == model_name]
            df_l = df_losses.loc[df_losses.model_name == model_name]

            acc = df_acc.loc[df_acc.result_of == 'train'].accuracy.values.tolist()
            val_acc = df_acc.loc[df_acc.result_of == 'validation'].accuracy.values.tolist()
            loss = df_l.loc[df_l.result_of == 'train'].loss.values.tolist()
            val_loss = df_l.loc[df_l.result_of == 'validation'].loss.values.tolist()
            epochs = len(df_acc)

            append_trace(model_name, acc, val_acc, loss, val_loss, epochs)

    else:
        for model_name, history in histories.items():
            acc = history['accuracy']
            val_acc = history['val_accuracy']
            loss = history['loss']
            val_loss = history['val_loss']
            epochs = list(range(1, len(acc) + 1))

            append_trace(model_name, acc, val_acc, loss, val_loss, epochs)
    fig['layout']['xaxis'].update(title='Epoch')
    fig['layout']['xaxis2'].update(title='Epoch')
    fig['layout']['yaxis'].update(title='Accuracy', range=[0, 1])
    fig['layout']['yaxis2'].update(title='Loss', range=[0, 1])

    iplot(fig, filename='accuracies-losses')


if __name__ == '__main__':
    plot_accuracy_and_loss()
