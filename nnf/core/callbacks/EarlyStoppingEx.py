"""EarlyStoppingEx to represent EarlyStoppingEx class."""
# -*- coding: utf-8 -*-
# Global Imports
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from warnings import warn as warning
import numpy as np

# Local Imports

class EarlyStoppingEx(EarlyStopping):
    """Stop training when a monitored quantity has stopped improving..

    Attributes
    ----------
        monitor: quantity to be monitored.
        <describe>.

        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        <describe>.

    wait : <describe>.
        <describe>.

    mode : one of {auto, min, max}
        In `min` mode, training will stop when the quantity monitored has stopped decreasing;
        in `max` mode it will stop when the quantity monitored has stopped increasing;
        in `auto` mode, the direction is automatically inferred from the name of the monitored quantity.

    Methods
    -------
    on_train_begin()
        <describe>.

    on_epoch_end()
        <describe>.

    Examples
    --------
    <describe>
    >>> nndb = EarlyStoppingEx(EarlyStopping)

    <describe>
    >>> nndb = EarlyStoppingEx(EarlyStopping)
    """

    def __init__(self, monitor='val_loss', patience=0, verbose=0, mode='auto'):
        """describe.

        Parameters
        ----------
        sel : selection structure
            Information to split the dataset.

        monitor : Describe
            describe.

        patience : Describe
            describe.

        verbose : Describe
            describe.

        mode : Describe
            describe.
        """
        super(EarlyStoppingEx, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.wait = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

    def on_train_begin(self, logs={}):
        """Construct a on_train_begin object.

        Parameters
        ----------
        sel : selection structure
            Information to split the dataset.

        logs : Describe
            describe.
        """
        self.wait = 0       # Allow instances to be re-used
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs={}):
        """Construct a on_epoch_end object.

        Parameters
        ----------
        sel : selection structure
            Information to split the dataset.

        logs : Describe
            describe.
        """
        current = logs.get(self.monitor)
        
        if current is None:
            warning('Early stopping requires %s available!' %
                          (self.monitor), RuntimeWarning)

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print('Epoch %05d: early stopping' % (epoch))
                self.model.stop_training = True
            self.wait += 1