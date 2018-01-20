# -*- coding: utf-8 -*-
"""
.. module:: Globals
   :platform: Unix, Windows
   :synopsis: Represent globals for nnf framework.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
import tensorflow as tf
import keras.backend as K

# Local Imports


class Metric:
    @staticmethod
    def r(y_true, y_pred):
        # This method will be called once from Keras, hence can initialize the local variables once only.
        _, update_op = tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true)
        K.get_session().run(tf.local_variables_initializer())
        return update_op

    @staticmethod
    def cov(y_true, y_pred):
        # This method will be called once from Keras, hence can initialize the local variables once only.
        _, update_op = tf.contrib.metrics.streaming_covariance(y_pred, y_true)
        K.get_session().run(tf.local_variables_initializer())
        return update_op

    @staticmethod
    def s_acc(y_true, y_pred):
        # This method will be called once from Keras, hence can initialize the local variables once only.
        _, update_op = tf.contrib.metrics.streaming_accuracy(y_pred, y_true)
        K.get_session().run(tf.local_variables_initializer())
        return update_op