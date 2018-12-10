"""The code was taken from carla-simulator/imitation-learning.

Code:
https://github.com/carla-simulator/imitation-learning/blob/master/agents/imitation/imitation_learning_network.py
This repository makes minor adjustment to this.
"""
from __future__ import print_function, unicode_literals

from future.builtins import object, range
import numpy as np
import tensorflow as tf

import constants as ilc

# from https://github.com/carla-simulator/imitation-learning/blob/62f93c2785a2452ca67eebf40de6bf33cea6cbce/agents/imitation/imitation_learning.py#L23  # noqa
DROPOUT_VEC_TRAIN = [1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 5
DROPOUT_VEC_INFER = [1.0 for _ in DROPOUT_VEC_TRAIN]


def weight_xavi_init(shape, name):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)


class Network(object):

    def __init__(self, dropout, image_shape, is_training):
        """ We put a few counters to see how many times we called each function """
        self._dropout_vec = dropout
        self._image_shape = image_shape
        self._is_training = is_training
        self._count_conv = 0
        self._count_pool = 0
        self._count_bn = 0
        self._count_activations = 0
        self._count_dropouts = 0
        self._count_fc = 0
        self._count_lstm = 0
        self._count_soft_max = 0
        self._conv_kernels = []
        self._conv_strides = []
        self._weights = {}
        self._features = {}

    """ Our conv is currently using bias """

    def conv(self, x, kernel_size, stride, output_size, padding_in='SAME'):
        self._count_conv += 1

        filters_in = x.get_shape()[-1]
        shape = [kernel_size, kernel_size, filters_in, output_size]

        weights = weight_xavi_init(shape, 'W_c_{}'.format(self._count_conv))
        bias = bias_variable([output_size], name='B_c_{}'.format(self._count_conv))

        self._weights['W_conv{}'.format(self._count_conv)] = weights
        self._conv_kernels.append(kernel_size)
        self._conv_strides.append(stride)

        conv_res = tf.add(tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding=padding_in,
                                       name='conv2d_{}'.format(self._count_conv)), bias,
                          name='add_{}'.format(self._count_conv))

        self._features['conv_block{}'.format(self._count_conv - 1)] = conv_res

        return conv_res

    def max_pool(self, x, ksize=3, stride=2):
        self._count_pool += 1
        return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1],
                              padding='SAME', name='max_pool{}'.format(self._count_pool))

    def bn(self, x):
        self._count_bn += 1
        return tf.contrib.layers.batch_norm(x, is_training=self._is_training,
                                            updates_collections=None,
                                            scope='bn{}'.format(self._count_bn))

    def activation(self, x):
        self._count_activations += 1
        return tf.nn.relu(x, name='relu{}'.format(self._count_activations))

    def dropout(self, x):
        self._count_dropouts += 1
        output = tf.nn.dropout(x, self._dropout_vec[self._count_dropouts - 1],
                               name='dropout{}'.format(self._count_dropouts))

        return output

    def fc(self, x, output_size):
        self._count_fc += 1
        filters_in = x.get_shape()[-1]
        shape = [filters_in, output_size]

        weights = weight_xavi_init(shape, 'W_f_{}'.format(self._count_fc))
        bias = bias_variable([output_size], name='B_f_{}'.format(self._count_fc))

        return tf.nn.xw_plus_b(x, weights, bias, name='fc_{}'.format(self._count_fc))

    def conv_block(self, x, kernel_size, stride, output_size, padding_in='SAME'):
        with tf.name_scope("conv_block{}".format(self._count_conv)):
            x = self.conv(x, kernel_size, stride, output_size, padding_in=padding_in)
            x = self.bn(x)
            x = self.dropout(x)

            return self.activation(x)

    def fc_block(self, x, output_size):
        with tf.name_scope("fc{}".format(self._count_fc + 1)):
            x = self.fc(x, output_size)
            x = self.dropout(x)
            self._features['fc_block{}'.format(self._count_fc + 1)] = x
            return self.activation(x)

    def get_weigths_dict(self):
        return self._weights

    def get_feat_tensors_dict(self):
        return self._features


def load_imitation_learning_network(input_image, input_data, mode):
    branches = []

    x = input_image
    if mode == tf.estimator.ModeKeys.TRAIN:
        dropout = DROPOUT_VEC_TRAIN
        is_training = True
    else:
        dropout = DROPOUT_VEC_INFER
        is_training = False

    with tf.name_scope('Network'):  # for a nicer Tensorboard graph, use: `with tf.variable_scope('Network'):`
        network_manager = Network(dropout, tf.shape(x), is_training)

        """conv1"""  # kernel sz, stride, num feature maps
        xc = network_manager.conv_block(x, 5, 2, 32, padding_in='VALID')
        xc = network_manager.conv_block(xc, 3, 1, 32, padding_in='VALID')

        """conv2"""
        xc = network_manager.conv_block(xc, 3, 2, 64, padding_in='VALID')
        xc = network_manager.conv_block(xc, 3, 1, 64, padding_in='VALID')

        """conv3"""
        xc = network_manager.conv_block(xc, 3, 2, 128, padding_in='VALID')
        xc = network_manager.conv_block(xc, 3, 1, 128, padding_in='VALID')

        """conv4"""
        xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID')
        xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID')
        """mp3 (default values)"""

        """ reshape """
        x = tf.reshape(xc, [-1, int(np.prod(xc.get_shape()[1:]))], name='reshape')

        """ fc1 """
        x = network_manager.fc_block(x, 512)
        """ fc2 """
        x = network_manager.fc_block(x, 512)

        """Process Control"""

        """ Speed (measurements)"""
        with tf.name_scope("Speed"):
            speed = input_data[1]  # get the speed from input data
            speed = network_manager.fc_block(speed, 128)
            speed = network_manager.fc_block(speed, 128)

        """ Joint sensory """
        j = tf.concat([x, speed], 1)
        j = network_manager.fc_block(j, 512)

        """Start BRANCHING"""
        branch_config = [
            [ilc.TGT_STEER, ilc.TGT_GAS, ilc.TGT_BRAKE],
            [ilc.TGT_STEER, ilc.TGT_GAS, ilc.TGT_BRAKE],
            [ilc.TGT_STEER, ilc.TGT_GAS, ilc.TGT_BRAKE],
            [ilc.TGT_STEER, ilc.TGT_GAS, ilc.TGT_BRAKE],
            [ilc.TGT_SPEED],
        ]
        for i in range(0, len(branch_config)):
            with tf.name_scope("Branch_{}".format(i)):
                if branch_config[i][0] == ilc.TGT_SPEED:
                    # we only use the image as input to speed prediction
                    branch_output = network_manager.fc_block(x, 256)
                    branch_output = network_manager.fc_block(branch_output, 256)
                else:
                    branch_output = network_manager.fc_block(j, 256)
                    branch_output = network_manager.fc_block(branch_output, 256)

                branches.append(network_manager.fc(branch_output, len(branch_config[i])))

        return branches
