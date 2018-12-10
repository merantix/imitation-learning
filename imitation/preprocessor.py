"""Converts data used in https://github.com/carla-simulator/imitation-learning from HDF5 files to tfrecords."""
from __future__ import unicode_literals

from collections import defaultdict
from glob import glob
import logging
import os
import random
import re

from future.builtins import range, zip
import h5py
import numpy as np
import tensorflow as tf

from common.util import tf_flags_util, tf_util
import constants as ilc

logging.basicConfig(level=logging.INFO)

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
tf.flags.DEFINE_string("preproc_config_paths", default="config-preprocess-debug.yaml",
                       help="Path to a YAML configuration files defining FLAG "
                            'values. Multiple files can be separated by commas. '
                            'Files are merged recursively. Setting a key in these '
                            'files is equivalent to setting the FLAG value with '
                            'the same name.')
flags.DEFINE_integer('h5_files_per_split', default=None,
                     help='Number of files to process for each of TRAIN and VAL.  Setting to None chooses all files.')
flags.DEFINE_string('data_dir',
                    default='/data/imitation_learning/h5_files/AgentHuman/',
                    help='Directory containing SeqTrain and SeqVal subdirs with hdf5 files.')
flags.DEFINE_string('preproc_output_dir',
                    default='/data/imitation_learning/preprocessed',
                    help='Directory to out TFRecords.')

tf_flags_util.overwrite_tf_flags_with_config(FLAGS, FLAGS.preproc_config_paths)

MAX_NUM_CLOUD_WORKERS = 10
WORKER_MACHINE_TYPE = 'n1-highcpu-2'

FILE_IDX_RE = re.compile('data_(\d+)\.h5')

COUNTERS = defaultdict(int)
COUNTERS['INTENTION_COUNTERS'] = defaultdict(int)

TRAIN_SPLIT = 'TRAIN'
VAL_SPLIT = 'VAL'

BATCHSIZE = 70  # this controls how many consecutive h5 files get combined to one tfrecord file


def get_files(subdir, data_dir, num_files=None):
    """Get list of files to preprocessor.

    Args:
        subdir: usually 'SeqVal' or 'SeqTrain'. distinguishes between train & test
        data_dir: parent directory of subdir
            (e.g. '/data/imitation_learning/h5_files/AgentHuman/')
        num_files: number of files to restrict this list to.  useful for local preprocessing

    Returns:
        list of h5 files to preprocess
    """
    h5_dir = os.path.join(data_dir, subdir)
    assert os.path.isdir(h5_dir), 'No directory {}'.format(h5_dir)

    h5_glob = os.path.join(h5_dir, '*.h5')
    h5_paths = glob(h5_glob)
    assert len(h5_paths) > 0, 'Found no files for glob {}'.format(h5_glob)
    logging.info("Found {} h5 files in {}".format(len(h5_paths), h5_dir))

    h5_paths = sorted(h5_paths)
    if num_files is not None:
        logging.info("Restricting to {} files from {}".format(num_files, h5_dir))
        h5_paths = h5_paths[:num_files]

    return h5_paths


def _get_key(filename, batch_idx):
    """Compute unique key for a data point."""
    basename = os.path.basename(filename)
    file_idx = FILE_IDX_RE.match(basename).group(1)
    assert batch_idx < 1000, 'increase digits in key for batch_idx {}'.format(batch_idx)
    key = 'f{}b{:03d}'.format(file_idx, batch_idx)
    return key


def read_h5_file(h5_file):
    """Read file in h5 format.

    Args:
        h5_file: fully qualified path to h5 file

    Returns:
        iterator over tf examples
    """
    global COUNTERS

    COUNTERS['H5_ATTEMPTED_COUNTER'] += 1
    try:
        # dict of format: {
        #   'rgb': <HDF5 dataset shape (batch_size, IMG_HEIGHT, IMG_WIDTH, 3), type "|u1",
        #   'targets': <HDF5 dataset shape (batch_size, 28), type "<f4">, }
        data = h5py.File(h5_file, 'r')

        rgb_dataset = data['rgb']
        targets_dataset = data['targets']

        batch_size = rgb_dataset.shape[0]
        for batch_idx in range(batch_size):
            features = {}

            # unique key
            key = _get_key(h5_file, batch_idx)  # unicode
            features[ilc.FEATKEY_KEY] = tf_util.bytes_feature(bytes(key))

            # image
            rgb = rgb_dataset[batch_idx, :, :, :]
            img_str = np.array(rgb, dtype=np.uint8).reshape(-1).tostring()
            features[ilc.FEATKEY_IMG] = tf_util.bytes_feature(img_str)

            # targets
            targets = targets_dataset[batch_idx, :]
            for tgt_key, tgt_val in zip(ilc.TGT_KEYS, targets):
                features[tgt_key] = tf_util.float_feature(tgt_val)

            tf_example = tf_util.make_tf_example(features)

            intention_idx = ilc.TGT_KEYS.index(ilc.TGT_HIGH_LVL_CMD)
            intention = targets[intention_idx]
            # a small percentage of the data has no intention; remove these examples
            if intention is None:
                COUNTERS['DROPPED_COUNTER'] += 1
                continue
            COUNTERS['INTENTION_COUNTERS'][int(intention)] += 1

            COUNTERS['TF_EXAMPLE_COUNTER'] += 1
            yield tf_example
    except Exception as e:  # noqa
        logging.error("Failure for file %s . Skipping.", h5_file)
        COUNTERS['H5_SKIP_COUNTER'] += 1


def write_tfrecord_file(output_filepath, some_h5_files):
    """Write tf.Examples given a list of h5_files.

    Args:
        output_filepath: str
        some_h5_files: List[str]
    """
    tf_record_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(output_filepath, options=tf_record_options)

    # Read a batch of h5 files
    for f in some_h5_files:
        tf_examples = list(read_h5_file(f))  # type: List[tf.Example]

        # Serialize to string
        tf_example_strs = map(lambda ex: ex.SerializeToString(), tf_examples)

        # Write
        for example_str in tf_example_strs:
            writer.write(example_str)

    writer.close()


def run_pipeline():
    for label, subdir in [(VAL_SPLIT, 'SeqVal'), (TRAIN_SPLIT, 'SeqTrain')]:

        # get list of h5 files to preprocess
        h5_files = get_files(subdir, data_dir=FLAGS.data_dir, num_files=FLAGS.h5_files_per_split)
        random.shuffle(h5_files)

        for index in range(0, len(h5_files), BATCHSIZE):

            output_filename = '{}-{:05d}.tfrecord.gz'.format(label, index / BATCHSIZE)
            output_filepath = os.path.join(FLAGS.preproc_output_dir, output_filename)

            upper_index = min(index + BATCHSIZE, len(h5_files))
            some_h5_files = h5_files[index:upper_index]

            write_tfrecord_file(output_filepath, some_h5_files)


def show_counters():
    for k, v in COUNTERS.items():
        logging.info('{key}: {counter}'.format(key=k, counter=dict(v) if isinstance(v, dict) else v))


def main():
    run_pipeline()
    show_counters()


if __name__ == '__main__':
    main()
