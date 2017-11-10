import os
import six
import csv
import numpy as np
from collections import Iterable, OrderedDict
from keras.callbacks import Callback
from keras import backend as K


class MyLearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs=None):
        optimizer = self.model.optimizer
        lr = optimizer.lr
        lr_d = lr
        if optimizer.initial_decay > 0:
            # lr *= (1. / (1. + self.decay * K.cast(self.iterations, K.dtype(self.decay))))
            lr_d *= (1. / (1. + optimizer.decay * K.cast(optimizer.iterations, K.dtype(optimizer.decay))))
        print('\nLR, LRd: {:.6f}, {:.6f}\n'.format(K.eval(lr), K.eval(lr_d)))


class MyCSVLogger(Callback):
    """Callback that streams epoch and batch results to a csv file.

    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.

    # Example
        ```python
        csv_logger = MyCSVLogger('training.log')
        model.fit(X_train, Y_train, callbacks=[csv_logger])
        ```

    # Arguments
        filename: filename of the csv file, e.g. 'run/log.csv'.
        separator: string used to separate elements in the csv file.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    """

    def __init__(self, filename, separator=',', append=False):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.file_flags = 'b' if six.PY2 and os.name == 'nt' else ''
        self.csv_file = None
        self.currentEpoch = 0
        self.currentBatch = 0
        self.currentLR = 0
        self.currentLRd = 0
        super(MyCSVLogger, self).__init__()

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            self.csv_file = open(self.filename, 'a' + self.file_flags)
        else:
            self.csv_file = open(self.filename, 'w' + self.file_flags)

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None

    def get_lr(self):
        optimizer = self.model.optimizer
        # TODO: if type(optimizer)...
        lr = optimizer.lr
        lr_d = lr
        if optimizer.initial_decay > 0:
            # lr *= (1. / (1. + self.decay * K.cast(self.iterations, K.dtype(self.decay))))
            lr_d *= (1. / (1. + optimizer.decay * K.cast(optimizer.iterations, K.dtype(optimizer.decay))))
        return [K.eval(lr), K.eval(lr_d)]

    def add_row(self, logs=None):
        logs = logs or {}

        def handle_value(d, k):
            k = d[k] if k in d else ""

            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict([(k, logs[k]) if k in logs else (k, 'NA') for k in self.keys])

        if not self.writer:
            self.keys = sorted(logs.keys())
            if "epoch" in self.keys:
                self.keys.remove("epoch")
            if "batch" in self.keys:
                self.keys.remove("batch")
            if "lr" in self.keys:
                self.keys.remove("lr")
            if "lr_d" in self.keys:
                self.keys.remove("lr_d")
            if "size" in self.keys:
                self.keys.remove("size")
            self.keys += ["val_" + k for k in self.keys]

            class CustomDialect(csv.excel):
                delimiter = self.sep

            cols = []
            cols += ['epoch', 'batch', 'lr', 'lr_d']
            cols += self.keys

            self.writer = csv.DictWriter(self.csv_file, fieldnames=cols, dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        # preapre row
        row_dict = OrderedDict({'epoch': self.currentEpoch,
                                'batch': self.currentBatch,
                                'lr': self.currentLR,
                                'lr_d': self.currentLRd})
        row_dict.update((key, handle_value(logs, key)) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_batch_begin(self, batch, logs=None):
        self.currentBatch = batch
        (self.currentLR, self.currentLRd) = self.get_lr()

    def on_batch_end(self, batch, logs=None):
        self.add_row(logs)

    def on_epoch_begin(self, epoch, logs=None):
        self.currentEpoch = epoch
        (self.currentLR, self.currentLRd) = self.get_lr()

    def on_epoch_end(self, epoch, logs=None):
        self.add_row(logs)
