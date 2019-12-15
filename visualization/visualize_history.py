import plotly.graph_objs as go
from plotly import subplots
from plotly.offline import iplot


def create_trace(x, y, ylabel, color):
    trace = go.Scatter(
        x=x, y=y,
        name=ylabel,
        marker=dict(color=color),
        mode="markers+lines",
        text=x
    )
    return trace


def plot_accuracy_and_loss(hist):
    acc = hist['accuracy']
    val_acc = hist['val_accuracy']
    loss = hist['loss']
    val_loss = hist['val_loss']
    epochs = list(range(1, len(acc) + 1))

    trace_ta = create_trace(epochs, acc, "Training accuracy", "Green")
    trace_va = create_trace(epochs, val_acc, "Validation accuracy", "Red")
    trace_tl = create_trace(epochs, loss, "Training loss", "Blue")
    trace_vl = create_trace(epochs, val_loss, "Validation loss", "Magenta")

    fig = subplots.make_subplots(rows=1, cols=2, subplot_titles=('Training and validation accuracy',
                                                                 'Training and validation loss'))
    fig.append_trace(trace_ta, 1, 1)
    fig.append_trace(trace_va, 1, 1)
    fig.append_trace(trace_tl, 1, 2)
    fig.append_trace(trace_vl, 1, 2)
    fig['layout']['xaxis'].update(title='Epoch')
    fig['layout']['xaxis2'].update(title='Epoch')
    fig['layout']['yaxis'].update(title='Accuracy', range=[0, 1])
    fig['layout']['yaxis2'].update(title='Loss', range=[0, 1])

    iplot(fig, filename='accuracy-loss')