# LR Range Test in Keras, by lgpang
# Save this script as lr_range.py

from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np
from utils import Logger


class BatchLearningRateScheduler(keras.callbacks.Callback):
    """Learning rate scheduler for each batch
    # Arguments
        schedule: a function that takes an batch index as input
            (integer, indexed from 0) and current learning rate
            and returns a new learning rate as output (float).
        verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, schedule, verbose=1):
        super(BatchLearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose
        self.global_batch = 0

    def on_batch_begin(self, batch, logs=None):
        self.global_batch += 1
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(K.get_value(self.model.optimizer.lr))
        try:  # new API
            lr = self.schedule(self.global_batch, lr)
        except TypeError:  # old API for backward compatibility
            lr = self.schedule(self.global_batch)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.lr, lr)

    def format_val(self, val):
        if isinstance(val, list) or isinstance(val, tuple):
            return "(%s)" % ", ".join([self.format_val(x) for x in val])
        else:
            return "%.6lf" % val

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        re = ", ".join(["%s: %s" % (key, self.format_val(value)) for (key, value) in logs.items()])
        Logger.log("Batch %06d: %s" % (self.global_batch + 1, re))


def LR_Range_Test_Schedule(batch):
    '''increase lr by a small amount per batch'''
    initial_lrate = 1.0E-7
    speed = 1.0E-7
    lr = initial_lrate + batch * speed
    return lr


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(LR_Range_Test_Schedule(len(self.losses)))

    #    Logger.log('batch %05d: loss = %lf, lr = %lf' % (batch + 1, logs.get('loss'), self.lr[-1]))


loss_history = LossHistory()
reduced_lr = BatchLearningRateScheduler(LR_Range_Test_Schedule)
callbacks_list = [loss_history, reduced_lr]
