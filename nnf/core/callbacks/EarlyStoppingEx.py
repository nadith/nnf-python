"""EarlyStoppingEx to represent EarlyStoppingEx class."""
# -*- coding: utf-8 -*-

# Global Imports
import warnings
import numpy as np
from nnf.keras.models import clone_model
from keras.callbacks import EarlyStopping

# Local Imports


class EarlyStoppingEx(EarlyStopping):
    """Stop training when a monitored quantity has stopped improving.

    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
    """

    def __init__(self, monitor='val_loss',
                 min_delta=0, patience=0, verbose=0, mode='auto'):
        super(EarlyStoppingEx, self).__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode)
        self._best_model = None

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
            return
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0

            # Save the model with best weights
            self._best_model = clone_model(self.model)
            self._best_model.set_weights(self.model.get_weights())

        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch

                # Load the best model saved
                self.model = self._best_model

                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))