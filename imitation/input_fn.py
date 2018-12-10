"""
Class for loading imitation learning TFRecords, and applying data augmentation during training.

imgaug code from Carla authors:
https://github.com/carla-simulator/imitation-learning/issues/1#issuecomment-355747357

Dosovitskiy et al. CARLA: An Open Urban Driving Simulator
http://proceedings.mlr.press/v78/dosovitskiy17a/dosovitskiy17a.pdf:
"To further reduce overfitting, we performed extensive data augmentation by adding Gaussian blur,
additive Gaussian noise, pixel dropout, additive and multiplicative brightness variation, contrast variation,
and saturation variation"

Codevilla et al. End-to-end Driving via Conditional Imitation Learning
http://vladlen.info/papers/conditional-imitation.pdf:
"Transformations include change in contrast, brightness, and tone, as well as addition of Gaussian blur, Gaussian noise,
salt-and-pepper noise, and region dropout (masking out a random set of rectangles in the image, each rectangle taking
roughly 1% of image area)"
"""
from __future__ import unicode_literals

import functools

import tensorflow as tf

from common.train import inputs
from common.util import img_aug
import constants as ilc


def convert_image_tf(img_str):
    rgb_image = tf.reshape(tf.decode_raw(img_str, tf.uint8), shape=(ilc.IMG_HEIGHT, ilc.IMG_WIDTH, 3))
    return rgb_image


def get_feat_schema():
    schema = {
        ilc.FEATKEY_KEY: tf.FixedLenFeature(dtype=tf.string, shape=[]),
        ilc.FEATKEY_IMG: tf.FixedLenFeature(dtype=tf.string, shape=[]),
    }
    for key in ilc.TGT_KEYS:
        schema[key] = tf.FixedLenFeature(dtype=tf.float32, shape=[])
    return schema


class Preprocessor(object):
    """Base class for preprocessing steps that run at training time, shortly before the data enters the model_fn.

    Raises:
        AssertionError if mode is note TRAIN
    """

    def preprocess(self, dataset, mode):
        """Applies transformation to dataset

        Args:
            dataset: a tf.data.Dataset
            mode: a tf.estimator.ModeKeys

        Returns:
            a tf.data.Dataset
        """
        raise NotImplementedError


class FilterValidIntention(Preprocessor):

    def preprocess(self, dataset, mode):
        dataset = dataset.filter(self.filter_valid_intentions)
        return dataset

    @staticmethod
    def filter_valid_intentions(tf_example):
        """Return True if high-level command is in {2, 3, 4, 5}.

        Args:
            tf_example: Dict[str, tf.Tensor]

        Returns:
            tf.Tensor (type=bool)
        """
        high_level_command = tf_example[ilc.TGT_HIGH_LVL_CMD]
        return tf.logical_and(
            tf.greater_equal(high_level_command, 2),
            tf.less_equal(high_level_command, 5))


class CarlaPreprocessor(Preprocessor):
    """Base class for preprocessing steps that run at training time, shortly before the data enters the model_fn."""

    def preprocess(self, dataset, mode):
        """Applies transformation to dataset

        Args:
            dataset: a tf.data.Dataset
            mode: a tf.estimator.ModeKeys

        Returns:
            a tf.data.Dataset
        """
        dataset = dataset.map(self.read_fn)
        return dataset

    @staticmethod
    def read_fn(tf_example):
        """Given a tf_example dict, separates into feature_dict and target_dict"""
        flat_img = tf_example[ilc.FEATKEY_IMG]
        img = convert_image_tf(flat_img)
        img = tf.cast(img, tf.float32)
        img = tf.squeeze(img)
        img = tf.div(img, 255.0)

        feats = {
            ilc.FEATKEY_IMG: img,
            ilc.TGT_SPEED: tf_example[ilc.TGT_SPEED],
        }
        tgts = {key: tf_example[key] for key in ilc.TGT_KEYS}

        return feats, tgts


class ProbabilisticImageAugmentor(Preprocessor):
    """Preprocessor that applies `image_transform` to input image with `augmentation_prob` probability.

    Raises:
        AssertionError if mode is not TRAIN
    """

    def __init__(self, augmentation_prob, image_transform):
        self.augmentation_prob = augmentation_prob
        self.image_transform = image_transform
        super(ProbabilisticImageAugmentor, self).__init__()

    def apply_to_image(self, feats, tgts):
        orig_img = feats[ilc.FEATKEY_IMG]
        weighted_coin = tf.less(tf.random_uniform([], 0, 1.0), self.augmentation_prob)
        img = tf.cond(
            weighted_coin,
            lambda: self.image_transform(orig_img),
            lambda: orig_img,
        )
        feats[ilc.FEATKEY_IMG] = img
        return feats, tgts

    def preprocess(self, dataset, mode):
        assert mode == tf.estimator.ModeKeys.TRAIN, 'Should not augment eval / inference'
        return dataset.map(self.apply_to_image)


def _rand_gauss_blur(img):
    stddev = tf.random_uniform([], 0, 1.5)
    return img_aug.gauss_blur(img, stddev=stddev)


def _rand_gauss_noise(img):
    sigma = tf.random_uniform([], 0, 0.05)
    coin = tf.less(tf.random_uniform([], 0, 1.0), 0.5)
    new_img = tf.cond(
        coin,
        lambda: img_aug.gauss_noise(img, True, stddev=sigma),
        lambda: img_aug.gauss_noise(img, False, stddev=sigma),
    )
    return new_img


def _rand_pixelwise_dropout(img):
    coin = tf.less(tf.random_uniform([], 0.0, 1.0), 0.5)
    p_pixel_drop = tf.random_uniform([], 0, 0.1)
    new_img = tf.cond(
        coin,
        lambda: img_aug.pixelwise_dropout(img, p_pixel_drop, True),
        lambda: img_aug.pixelwise_dropout(img, p_pixel_drop, False),
    )
    return new_img


def _rand_coarse_pixelwise_dropout(img):
    coin = tf.less(tf.random_uniform([], 0.0, 1.0), 0.5)
    p_pixel_drop = tf.random_uniform([], 0, 0.1)
    p_height = tf.random_uniform([], 0.08, 0.2)
    p_width = tf.random_uniform([], 0.08, 0.2)
    new_img = tf.cond(
        coin,
        lambda: img_aug.coarse_pixelwise_dropout(img, p_height, p_width, p_pixel_drop, True),
        lambda: img_aug.coarse_pixelwise_dropout(img, p_height, p_width, p_pixel_drop, False),
    )
    return new_img


def train_input_fn(tfrecord_fpaths, batch_size, shuffle_buffer_size):
    input_fn = inputs.input_fn_factory(
        tfrecord_fpaths=tfrecord_fpaths,
        feature_schema=get_feat_schema(),
        batch_size=batch_size,
        mode=tf.estimator.ModeKeys.TRAIN,
        num_epochs=None,
        model_preprocessors=[
            FilterValidIntention(),
            CarlaPreprocessor(),
            ProbabilisticImageAugmentor(0.09, _rand_gauss_blur),
            ProbabilisticImageAugmentor(0.09, _rand_gauss_noise),
            ProbabilisticImageAugmentor(0.30, _rand_pixelwise_dropout),
            ProbabilisticImageAugmentor(0.30, _rand_coarse_pixelwise_dropout),
            ProbabilisticImageAugmentor(0.30, functools.partial(tf.image.random_brightness, max_delta=32 / 255)),
            ProbabilisticImageAugmentor(0.30, functools.partial(tf.image.random_saturation, lower=0.5, upper=1.5)),
            ProbabilisticImageAugmentor(0.09, functools.partial(tf.image.random_contrast, lower=0.5, upper=1.2)),
        ],
        shuffle=True,
        shuffle_buffer_size=shuffle_buffer_size,
    )
    return input_fn


def evaluation_input_fn(tfrecord_fpaths, batch_size):
    input_fn = inputs.input_fn_factory(
        tfrecord_fpaths=tfrecord_fpaths,
        feature_schema=get_feat_schema(),
        batch_size=batch_size,
        mode=tf.estimator.ModeKeys.EVAL,
        num_epochs=1,
        model_preprocessors=[FilterValidIntention(), CarlaPreprocessor()],
    )
    return input_fn
