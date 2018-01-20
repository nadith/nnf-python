# -*- coding: utf-8 -*-
"""
.. module:: TensorBoardEx
   :platform: Unix, Windows
   :synopsis: Represent extended TensorBoard class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from keras.callbacks import TensorBoard
from keras import backend as K
import tensorflow as tf
import numpy as np

# Local Imports


class TensorBoardEx(TensorBoard):
    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None):
        self.val_iter = None
        self.validation_steps = None
        super(TensorBoardEx, self).__init__(
                    log_dir,
                    histogram_freq,
                    batch_size,
                    write_graph,
                    write_grads,
                    write_images,
                    embeddings_freq,
                    embeddings_layer_names,
                    embeddings_metadata)

    def init(self, val_gen, validation_steps):
        self.val_iter = val_gen.core_iter.clone()
        self.val_iter.batch_size = self.batch_size
        self.validation_steps = validation_steps

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if self.histogram_freq and (not self.val_iter and not self.validation_data):
            raise ValueError('If printing histograms, validation_data must be provided')

        if self.histogram_freq and (self.val_iter or self.validation_data):
            if epoch % self.histogram_freq == 0:

                val_data = self.validation_data
                tensors = (self.model.inputs +
                           self.model.targets +
                           self.model.sample_weights)

                if self.model.uses_learning_phase:
                    tensors += [K.learning_phase()]

                # Validation sample count
                if val_data is None:
                    val_size = self.val_iter.N
                else:
                    assert len(val_data) == len(tensors)
                    val_size = val_data[0].shape[0]

                # Exhaust the generator batch by batch
                i = 0
                count = 0
                while i < val_size and count < self.validation_steps:
                    step = min(self.batch_size, val_size - i)

                    if val_data is None:
                        if self.model.uses_learning_phase: # when dropout layers are used (turn off in validation)
                            batch_val = list(next(self.val_iter))
                            inputs, outputs = batch_val
                            inputs = inputs if isinstance(inputs, list) else [inputs]
                            outputs = outputs if isinstance(outputs, list) else [outputs]
                            outputs_sample_weights = [np.ones(output.shape[0]) for output in outputs] # Adding sample_weights
                            batch_val = inputs + outputs + outputs_sample_weights
                            batch_val.append(0.0)  # To match len(tensors); placeholder value for learning_phase, 0.0 means testing, 1.0 means training
                        else:
                            batch_val = list(next(self.val_iter))
                            inputs, outputs = batch_val
                            inputs = inputs if isinstance(inputs, list) else [inputs]
                            outputs = outputs if isinstance(outputs, list) else [outputs]
                            outputs_sample_weights = [np.ones(output.shape[0]) for output in outputs] # Adding sample_weights
                            batch_val = inputs + outputs + outputs_sample_weights
                    else:
                        if self.model.uses_learning_phase:  # when dropout layers are used (turn off in validation)
                            # do not slice the learning phase
                            batch_val = [x[i:i + step] for x in val_data[:-1]]
                            batch_val.append(val_data[-1])
                        else:
                            batch_val = [x[i:i + step] for x in val_data]

                    assert len(batch_val) == len(tensors)
                    feed_dict = dict(zip(tensors, batch_val))
                    result = self.sess.run([self.merged], feed_dict=feed_dict)
                    summary_str = result[0]
                    self.writer.add_summary(summary_str, epoch)
                    i += self.batch_size
                    count += 1

        if self.embeddings_freq and self.embeddings_ckpt_path:
            if epoch % self.embeddings_freq == 0:
                self.saver.save(self.sess,
                                self.embeddings_ckpt_path,
                                epoch)

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
        self.writer.flush()