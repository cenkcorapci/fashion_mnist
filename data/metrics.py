import logging
from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

from config import TB_LOGS_PATH


def get_tensorboard_scalars(tensorboard_directory=TB_LOGS_PATH):
    accuracies = []
    losses = []
    for filename in Path(tensorboard_directory).rglob('events.out.tfevents.*'):
        splitted = str(filename).split('/')
        result_of, model_name = splitted[-2], splitted[-3]
        try:
            ea = event_accumulator.EventAccumulator(str(filename),
                                                    size_guidance={  # see below regarding this argument
                                                        event_accumulator.SCALARS: 0
                                                    })
            ea.Reload()
            for e in ea.Scalars('epoch_accuracy'):
                accuracies.append([model_name, result_of, e.step, e.value])
            for l in ea.Scalars('epoch_loss'):
                losses.append([model_name, result_of, l.step, l.value])
        except Exception as exp:
            logging.error(exp)
    df_accuracies = pd.DataFrame(accuracies)
    df_accuracies.columns = ['model_name', 'result_of', 'step', 'accuracy']
    df_losses = pd.DataFrame(losses)
    df_losses.columns = ['model_name', 'result_of', 'step', 'loss']
    return df_accuracies, df_losses
