import random

import matplotlib
import plotly.graph_objs as go
from plotly import subplots
from plotly.offline import iplot

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


def plot_accuracy_and_loss(histories):
    fig = subplots.make_subplots(rows=2, cols=2, subplot_titles=('Training accuracy', 'Validation accuracy',
                                                                 'Training loss ', 'Validation loss'))
    for model_name, history in histories.items():
        acc = history['accuracy']
        val_acc = history['val_accuracy']
        loss = history['loss']
        val_loss = history['val_loss']
        epochs = list(range(1, len(acc) + 1))
        color = random.choice(hex_colors_only)

        trace_ta = create_trace(epochs, acc, model_name, color)
        trace_va = create_trace(epochs, val_acc, model_name, color)
        trace_tl = create_trace(epochs, loss, model_name, color)
        trace_vl = create_trace(epochs, val_loss, model_name, color)

        fig.append_trace(trace_ta, 1, 1)
        fig.append_trace(trace_va, 1, 2)
        fig.append_trace(trace_tl, 2, 1)
        fig.append_trace(trace_vl, 2, 2)
    fig['layout']['xaxis'].update(title='Epoch')
    fig['layout']['xaxis2'].update(title='Epoch')
    fig['layout']['yaxis'].update(title='Accuracy', range=[0, 1])
    fig['layout']['yaxis2'].update(title='Loss', range=[0, 1])

    iplot(fig, filename='accuracy-loss')
