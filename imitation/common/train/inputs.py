from __future__ import absolute_import, division, unicode_literals
from future.utils import string_types
from glob import glob

import tensorflow as tf


def input_fn_factory(tfrecord_fpaths,
                     feature_schema,
                     batch_size,
                     mode,
                     num_epochs=None,
                     shuffle=False,
                     model_preprocessors=None,
                     shuffle_buffer_size=1000):
    """
    Args:
        tfrecord_fpaths: A list of tfrecord paths.
        feature_schema: feature schema for decoding tfrecords
        batch_size: number of examples per batch
        mode: one tf.estimator.ModeKeys.TRAIN/EVAL/PREDICT
        num_epochs: number of epochs to pass over dataset.  if None, continue indefinitely
        model_preprocessors: list of functions that take (features, labels) and return (features, labels)
        shuffle: should examples be randomized?
        shuffle_buffer_size: size of shuffle buffer. larger size improves pseudo-randomness, but increases startup time

    Returns:
        a (factory) function that itself returns batches of (features, labels).  The returned function should be called
        during a graph construction.
    """
    # check that files are unique
    assert len(tfrecord_fpaths) == len(set(tfrecord_fpaths)), 'tfrecord_paths {} are not unique'.format(tfrecord_fpaths)
    assert len(tfrecord_fpaths) > 0, 'No tfrecords specified'
    assert not shuffle or shuffle_buffer_size is not None, 'If using shuffle, please specify shuffle buffer size'

    def _inner():
        """Read tfrecords, this func will be returned"""
        dataset = tf.data.Dataset.from_tensor_slices(tfrecord_fpaths)
        if shuffle:
            dataset = dataset.shuffle(len(tfrecord_fpaths))
        dataset = dataset.flat_map(lambda filename: tf.data.TFRecordDataset(filename, compression_type="GZIP"))
        # deserialize tfexamples
        dataset = dataset.map(lambda serialized: tf.parse_single_example(serialized, feature_schema))

        for preprocessor in model_preprocessors or []:
            dataset = preprocessor.preprocess(dataset, mode)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        example_batch = iterator.get_next()
        return example_batch

    return _inner


def get_tfrecord_paths(globs):
    if isinstance(globs, string_types):
        globs = [globs]

    assert len(globs) == len(set(globs)), 'Specified globs are not unique: {}'.format(globs)

    paths = []
    for glob_ in globs:
        paths.extend(glob(glob_))

    assert len(paths) > 0, 'No tfrecords found for glob(s) {}'.format(globs)
    assert len(paths) == len(set(paths)), 'Globs resulted in non-unique set of paths.\nGlobs: {}\nPaths: {}'.format(
        globs, paths)
    sorted_unique_paths = sorted(set(paths))
    return sorted_unique_paths
