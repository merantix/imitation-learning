#!/usr/bin/env python
from __future__ import unicode_literals

import importlib
import logging
import os
import shutil

import tensorflow as tf

from common.train import inputs
from common.util import file_util, tf_flags_util
import constants as ilc
import input_fn
from evaluation import EvalHook, get_eval_metrics

logging.basicConfig(level=logging.INFO)

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
tf.flags.DEFINE_string("config_paths", default="config-train-debug.yaml",
                       help="Path to a YAML configuration files defining FLAG "
                            "values. Multiple files can be separated by commas. "
                            "Files are merged recursively. Setting a key in these "
                            "files is equivalent to setting the FLAG value with "
                            "the same name.")
tf.flags.DEFINE_string("model_type", default="conditional_il_learning_model",
                       help="Model that will be used. "
                            "Specify a module name that exists in models package.")
flags.DEFINE_float('learning_rate', default=0.0002, help='Learning rate for ADAM optimizer.')
flags.DEFINE_integer('shuffle_buffer_size', default=1000,
                     help='Size of shuffle buffer (larger buffer increases randomness).')
flags.DEFINE_integer('save_summary_steps', default=100,
                     help='Number of steps between tensorboard summary saves.')

# The following default values were chosen based on empirically evaluating the scales of steer, gas & brake
flags.DEFINE_float('lambda_gas', default=1., help='Weight in loss for gas.')
flags.DEFINE_float('lambda_steer', default=8.571, help='Weight in loss for steering.')
flags.DEFINE_float('lambda_brake', default=1., help='Weight in loss for brake.')
flags.DEFINE_float('lambda_speed', default=1., help='Weight in loss for speed.')

flags.DEFINE_string('output_dir', default='/data/imitation_learning/experiments/local_test',
                    help='Output directory used to store train and eval output.')
flags.DEFINE_string('train_path', default='/data/imitation_learning/preprocessed/TRAIN*.tfrecord.gz',
                    help='List of paths containing the training data to use, as '
                         'glob-style patterns.')
flags.DEFINE_string('validation_path', default='/data/imitation_learning/preprocessed/VAL*.tfrecord.gz',
                    help='List of paths containing the validation data to use, as '
                         'glob-style patterns.')
flags.DEFINE_integer('train_steps', default=100,
                     help='Number of training steps (i.e., batches) to run. This '
                          'count is global, so if we already ran N steps in a saved '
                          'checkpoint, then only (train_steps - N) more steps will '
                          'be run. Furthermore, training might terminate early if '
                          'you include a monitor that implements early termination.')
flags.DEFINE_integer('train_batch_size', default=1,
                     help='Batch size for training. Affects gradient updates and '
                          'execution performance.')
flags.DEFINE_integer('save_checkpoints_steps', default=1000,
                     help='Number of training steps between checkpoint saves.  '
                          'Checkpoints are used by Tensorflow monitors that perform '
                          'evaluation runs, check early termination conditions, '
                          'etc.  An evaluation is performed each time a checkpoint '
                          'is saved.')
flags.DEFINE_integer('eval_batch_size', default=1,
                     help='Batch size of train/val examples during evaluation mode. '
                          'Only affects performance; generally we want the largest '
                          'size that still fits in memory.')

flags.DEFINE_integer('learning_rate_decay_steps', default=50000,
                     help='Learning rate decay steps. '
                          'Will be used by the function: tf.train.exponential_decay')
flags.DEFINE_float('learning_rate_decay_factor', default=0.5,
                   help='Learning rate decay rate.'
                        'Will be used by the function: tf.train.exponential_decay')
flags.DEFINE_float('adam_beta1', default=0.7,
                   help='beta1 in Adam optimizer'
                        'The exponential decay rate for the 1st moment estimates.')
flags.DEFINE_float('adam_beta2', default=0.85,
                   help='beta2 in Adam optimizer'
                        'The exponential decay rate for the 2nd moment estimates.')

tf_flags_util.overwrite_tf_flags_with_config(FLAGS, FLAGS.config_paths)


_module = importlib.import_module('models.{}'.format(FLAGS.model_type))
try:
    model = _module.model
except AttributeError:
    logging.error("Module {} must contain a function called model()".format(str(_module)))
    raise


def train_input_fn():
    train_files = inputs.get_tfrecord_paths(FLAGS.train_path)
    return input_fn.train_input_fn(
        tfrecord_fpaths=train_files,
        batch_size=FLAGS.train_batch_size,
        shuffle_buffer_size=FLAGS.shuffle_buffer_size,
    )


def eval_validation_input_fn():
    """Evaluate on validation set."""
    val_files = inputs.get_tfrecord_paths(FLAGS.validation_path)
    return input_fn.evaluation_input_fn(
        tfrecord_fpaths=val_files,
        batch_size=FLAGS.eval_batch_size,
    )


def eval_train_input_fn():
    """Evaluate on train set."""
    train_val_files = inputs.get_tfrecord_paths(FLAGS.train_path)
    return input_fn.evaluation_input_fn(
        tfrecord_fpaths=train_val_files,
        batch_size=FLAGS.eval_batch_size,
    )


def model_fn(features, labels, mode):
    """Build model function.

    Args:
        features: Dict[str, tf.Tensor]
            'image': (?, 88, 200, 3)
            'Speed': (?, )
        labels: Dict[str, tf.Tensor]
            The dict keys are the strings listed in ilc.TGT_KEYS
        mode: tf.estimator.ModeKeys
    Returns:
        tf.estimator.EstimatorSpec
    """
    loss_weights = {
        ilc.TGT_STEER: FLAGS.lambda_steer,
        ilc.TGT_GAS: FLAGS.lambda_gas,
        ilc.TGT_BRAKE: FLAGS.lambda_brake,
        ilc.TGT_SPEED: FLAGS.lambda_speed,
    }
    total_loss, selected_pred, predictions = model(features, labels, mode, loss_weights)

    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        with tf.name_scope('eval_metrics'):
            eval_metric_ops = get_eval_metrics(selected_pred=selected_pred,
                                               labels=labels,
                                               loss_weights=loss_weights)
    elif mode == tf.estimator.ModeKeys.PREDICT:
        eval_metric_ops = None
    else:
        raise NameError('No such mode {}'.format(mode))

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                                                   global_step,
                                                   FLAGS.learning_rate_decay_steps,
                                                   FLAGS.learning_rate_decay_factor,
                                                   staircase=True,
                                                   name='learning_rate_decay')
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=FLAGS.adam_beta1,
                                           beta2=FLAGS.adam_beta2,
                                           name='AdamOptimizer')
        train_op = optimizer.minimize(total_loss, global_step=tf.train.get_or_create_global_step())
    elif mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]:
        train_op = None
    else:
        raise NameError('No such mode {}'.format(mode))

    return tf.estimator.EstimatorSpec(
        mode, predictions=predictions, loss=total_loss, train_op=train_op, eval_metric_ops=eval_metric_ops,
    )


class ImitationLearningTrainer(object):
    """Implements interface for training_deployment via run_train.py"""
    def __init__(self):
        logging.info("Creating estimator")
        estimator_config = tf.estimator.RunConfig(
            save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            save_summary_steps=FLAGS.save_summary_steps,
            keep_checkpoint_max=None,  # keep all checkpoints
        )
        self.estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=FLAGS.output_dir,
            config=estimator_config,
        )

    def execute(self):
        """Runs trainer module locally, based on config params"""
        file_util.create_directory(FLAGS.output_dir)
        self.train()

    def deploy(self):
        """Executes a deployer in the correct environment."""
        self.execute()

    def train(self):
        """Runs training based on mparam config"""
        if os.path.isdir(FLAGS.output_dir):
            logging.info("Deleting existing dir {}".format(FLAGS.output_dir))
            shutil.rmtree(FLAGS.output_dir)

        logging.info("Running train")
        self.estimator.train(
            input_fn=train_input_fn(),
            max_steps=FLAGS.train_steps,
            saving_listeners=[
                EvalHook(self.estimator, eval_validation_input_fn(), 'validation'),
                EvalHook(self.estimator, eval_train_input_fn(), 'train_val'),
            ],
        )


def get_model_dir():
    """
    Returns:
      Output directory used to store train output.
    """
    return os.path.join(FLAGS.output_dir, 'train')


def main():
    file_util.create_directory(get_model_dir())

    training_deployer = ImitationLearningTrainer()
    training_deployer.deploy()


if __name__ == '__main__':
    main()
