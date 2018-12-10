"""Utils related to tensorflow."""
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf


def _prepare_value_for_encoding_in_tf_example(value):
    if type(value) is np.ndarray:
        value = list(value.reshape(-1))
    elif not isinstance(value, list):
        value = [value]
    return value


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    value = _prepare_value_for_encoding_in_tf_example(value)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    value = _prepare_value_for_encoding_in_tf_example(value)
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def make_tf_example(features):
    """Wrapper for converting feature_dict into a tf_example"""
    tf_example = tf.train.Example(
        features=tf.train.Features(feature=features))
    return tf_example
