#!/usr/bin/env python
from __future__ import unicode_literals
import logging

import tensorflow as tf
from tensorflow.metrics import mean_absolute_error, mean_squared_error

import constants as ilc

logging.basicConfig(level=logging.INFO)


class EvalHook(tf.train.CheckpointSaverListener):
    def __init__(self, estimator, eval_input_fn, name):
        super(EvalHook, self).__init__()
        self.estimator = estimator
        self.input_fn = eval_input_fn
        self.name = name

    def after_save(self, session, global_step_value):
        logging.info('running eval on {} for step {}'.format(self.name, global_step_value))
        self.estimator.evaluate(
            self.input_fn, name=self.name)


def get_eval_metrics(selected_pred, labels, loss_weights):
    """Get evaluation metrics.

    Args:
        selected_pred: Dict[str, tf.Tensor]
            'steer': (?, )
            'gas': (?, )
            'brake': (?, )
            'speed': (?, )
        labels: Dict[str, tf.Tensor]
            'steer': (?, )
            'gas': (?, )
            'brake': (?, )
            'speed': (?, )
        loss_weights: Dict[str, float]

    Returns: evaluation_metrics
        Dict[str, Tuple]
    """
    batch_size = tf.shape(labels[ilc.TGT_STEER])[0]

    labels_tensor = tf.stack([labels[k] for k in ilc.OUTPUT_KEYS_AND_SPEED], axis=1)  # shape: (?, 3)
    selected_pred_tensor = tf.stack([selected_pred[k] for k in ilc.OUTPUT_KEYS_AND_SPEED], axis=1)  # shape: (?, 3)

    weights = tf.stack([
        tf.fill([batch_size], loss_weights[ilc.TGT_STEER]),
        tf.fill([batch_size], loss_weights[ilc.TGT_GAS]),
        tf.fill([batch_size], loss_weights[ilc.TGT_BRAKE]),
        tf.fill([batch_size], loss_weights[ilc.TGT_SPEED]),
    ], axis=1)

    with tf.name_scope('mae'):
        mae_steer = mean_absolute_error(labels[ilc.TGT_STEER], selected_pred[ilc.TGT_STEER])  # type: tuple
        mae_gas = mean_absolute_error(labels[ilc.TGT_GAS], selected_pred[ilc.TGT_GAS])
        mae_brake = mean_absolute_error(labels[ilc.TGT_BRAKE], selected_pred[ilc.TGT_BRAKE])
        mae_speed = mean_absolute_error(labels[ilc.TGT_SPEED], selected_pred[ilc.TGT_SPEED])

        nonspeed_mae = mean_absolute_error(labels_tensor[:, :3], selected_pred_tensor[:, :3], weights=weights[:, :3],
                                           name='nonspeed_loss')
        speed_mae = mean_absolute_error(labels[ilc.TGT_SPEED], selected_pred[ilc.TGT_SPEED], name='speed_loss')

        mae_total = mean_absolute_error(labels_tensor, selected_pred_tensor, weights=weights, name='mae_total')

    with tf.name_scope('mse'):
        mse_steer = mean_squared_error(labels[ilc.TGT_STEER], selected_pred[ilc.TGT_STEER])
        mse_gas = mean_squared_error(labels[ilc.TGT_GAS], selected_pred[ilc.TGT_GAS])
        mse_brake = mean_squared_error(labels[ilc.TGT_BRAKE], selected_pred[ilc.TGT_BRAKE])
        mse_speed = mean_squared_error(labels[ilc.TGT_SPEED], selected_pred[ilc.TGT_SPEED])

        nonspeed_mse = mean_squared_error(labels_tensor[:, :3], selected_pred_tensor[:, :3], weights=weights[:, :3],
                                          name='nonspeed_loss')
        speed_mse = mean_squared_error(labels[ilc.TGT_SPEED], selected_pred[ilc.TGT_SPEED], name='speed_loss')

        mse_total = mean_squared_error(labels_tensor, selected_pred_tensor, weights=weights, name='mse_total')

    with tf.name_scope('predictions'):
        pred_steer = tf.reshape(labels[ilc.TGT_STEER], shape=(batch_size,))  # shape: (?, )
        pred_gas = tf.reshape(labels[ilc.TGT_GAS], shape=(batch_size,))  # shape: (?, )
        pred_brake = tf.reshape(labels[ilc.TGT_BRAKE], shape=(batch_size,))  # shape: (?, )
        pred_speed = tf.reshape(labels[ilc.TGT_SPEED], shape=(batch_size,))  # shape: (?, )

        pred_steer = tf.metrics.mean(tf.abs(pred_steer))
        pred_gas = tf.metrics.mean(pred_gas)
        pred_brake = tf.metrics.mean(pred_brake)
        pred_speed = tf.metrics.mean(pred_speed)

    with tf.name_scope('ground_truth'):
        gt_steer = tf.reshape(labels[ilc.TGT_STEER], shape=(batch_size, ))  # shape: (?, )
        gt_gas = tf.reshape(labels[ilc.TGT_GAS], shape=(batch_size, ))  # shape: (?, )
        gt_brake = tf.reshape(labels[ilc.TGT_BRAKE], shape=(batch_size, ))  # shape: (?, )
        gt_speed = tf.reshape(labels[ilc.TGT_SPEED], shape=(batch_size,))  # shape: (?, )

        gt_steer = tf.metrics.mean(tf.abs(gt_steer))
        gt_gas = tf.metrics.mean(gt_gas)
        gt_brake = tf.metrics.mean(gt_brake)
        gt_speed = tf.metrics.mean(gt_speed)

    evaluation_metrics = {
        'mae_total': mae_total,
        'mae_steer': mae_steer,
        'mae_gas': mae_gas,
        'mae_brake': mae_brake,
        'mae_speed': mae_speed,
        'speed_mae': speed_mae,
        'nonspeed_mae': nonspeed_mae,

        'mse_total': mse_total,
        'mse_steer': mse_steer,
        'mse_gas': mse_gas,
        'mse_brake': mse_brake,
        'mse_speed': mse_speed,
        'speed_mse': speed_mse,
        'nonspeed_mse': nonspeed_mse,

        'pred_steer': pred_steer,
        'pred_gas': pred_gas,
        'pred_brake': pred_brake,
        'pred_speed': pred_speed,

        'gt_steer': gt_steer,
        'gt_gas': gt_gas,
        'gt_brake': gt_brake,
        'gt_speed': gt_speed,
    }
    return evaluation_metrics
