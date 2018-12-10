#!/usr/bin/env python
from __future__ import unicode_literals

import tensorflow as tf

import constants as ilc
from models.imitation_learning_network import load_imitation_learning_network


def model(features, labels, mode, loss_weights):
    """Construct the model.

    Args:
        features: Dict[str, tf.Tensor]
            'image': (?, 88, 200, 3)
            'Speed': (?, )
        labels: Dict[str, tf.Tensor]
            The dict keys are the strings listed in ilc.TGT_KEYS
        mode: tf.estimator.ModeKeys
        loss_weights: Dict[str, float]
            keys: 'steer', 'gas', 'brake', 'speed'

    Returns:
        total_loss: tf.Tensor
            shape: ()
        selected_pred: Dict[str, tf.Tensor]
            'steer': shape: (?, )
            'gas': shape: (?, )
            'brake': shape: (?, )
            'speed': shape: (?, )
        predictions: Dict[str, tf.Tensor]
            'Branches': shape: (?, 4, 3)
            'Speed_Branch': shape: (?, )
    """
    image = features[ilc.FEATKEY_IMG]
    batch_size = tf.shape(features[ilc.TGT_SPEED])[0]

    # Note: this is reverse engineered to work with the imitation-learning network
    speed_pair = [None, tf.reshape(features[ilc.TGT_SPEED], shape=(batch_size, 1))]
    branches = load_imitation_learning_network(image, speed_pair, mode)  # type: List[tf.Tensor]

    # The last branch is speed, we want to exclude this
    pred_all_branches = tf.stack(branches[:4], axis=1)  # shape: (?, 4, 3)
    pred_speed = branches[ilc.SPEED_BRANCH_INDEX][:, 0]  # shape: (?, )

    predictions = {
        ilc.OUTPUT_BRANCHES: pred_all_branches,
        ilc.OUTPUT_BRANCH_SPEED: pred_speed,
    }

    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:

        # calculate ground truth for conditional branch
        labels_tensor = tf.stack([labels[k] for k in ilc.OUTPUT_KEYS_AND_SPEED], axis=1)  # shape: (?, 4)

        with tf.name_scope('predictions'):
            # Compute loss against output head for this high level command.

            # high_level_cmds are [2, 3, 4, 5]; convert to 0-indexing
            branch_idx = tf.cast(labels[ilc.TGT_HIGH_LVL_CMD], tf.int32) - 2  # shape: (?,)

            # [[0, branch_idx_0], [1, branch_idx_1], ...]
            batch_idx_branch_idx = tf.stack([tf.range(batch_size), branch_idx], axis=1)  # shape: (?, 2)

            # While pred_all_branches holds the predictions for each of the 4 branches,
            # the selected predictions will hold the predictions for the branch specified in the high-level command.
            # More precisely speaking, the tf.gather_nd operation picks only selected branches for each sample.

            # [(steer_0, gas_0, brake_0), (steer_1, gas_1, brake_1), ...]
            selected_pred_tensor = tf.gather_nd(params=pred_all_branches, indices=batch_idx_branch_idx)  # shape: (?, 3)

        with tf.name_scope('loss'):
            weights = tf.stack([
                tf.fill([batch_size], loss_weights[ilc.TGT_STEER]),
                tf.fill([batch_size], loss_weights[ilc.TGT_GAS]),
                tf.fill([batch_size], loss_weights[ilc.TGT_BRAKE]),
                tf.fill([batch_size], loss_weights[ilc.TGT_SPEED]),
            ], axis=1)  # shape: (?, 3)  # size should be the same as selected_pred_tensor

            pred = tf.concat([selected_pred_tensor, tf.reshape(pred_speed, shape=(batch_size, 1))], axis=1)
            total_loss = tf.losses.mean_squared_error(labels_tensor, pred, weights=weights, scope='total_loss')
            tf.summary.scalar("Loss", total_loss)

        selected_pred = {
            ilc.TGT_STEER: selected_pred_tensor[:, 0],
            ilc.TGT_GAS: selected_pred_tensor[:, 1],
            ilc.TGT_BRAKE: selected_pred_tensor[:, 2],
            ilc.TGT_SPEED: predictions[ilc.OUTPUT_BRANCH_SPEED],
        }

        return total_loss, selected_pred, predictions

    elif mode == tf.estimator.ModeKeys.PREDICT:
        return None, None, predictions
    else:
        raise NameError('No such mode {}'.format(mode))
