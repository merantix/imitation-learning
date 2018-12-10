#!/usr/bin/env python
"""This idea is taken from:
- https://github.com/google/seq2seq/blob/master/bin/infer.py
- https://github.com/google/seq2seq/blob/master/bin/train.py
"""

from __future__ import unicode_literals

import os
import yaml

import tensorflow as tf
from tensorflow import gfile


def _deep_merge_dict(dict_x, dict_y, path=None):
    """Recursively merges dict_y into dict_x.

    Adapted from
    https://github.com/google/seq2seq/blob/master/seq2seq/configurable.py#L69
    """
    if path is None:
        path = []
    for key in dict_y:
        if key in dict_x:
            if isinstance(dict_x[key], dict) and isinstance(dict_y[key], dict):
                _deep_merge_dict(dict_x[key], dict_y[key], path + [str(key)])
            elif dict_x[key] == dict_y[key]:
                pass  # same leaf value
            else:
                dict_x[key] = dict_y[key]
        else:
            dict_x[key] = dict_y[key]
    return dict_x


def overwrite_tf_flags_with_config(flags, config_paths):
    """Load flags from config file

    Adapted from:
    https://github.com/google/seq2seq/blob/7f485894d412e8d81ce0e07977831865e44309ce/bin/train.py#L244
    """
    final_config = {}
    if not config_paths:
        return
    for config_path in config_paths.split(","):
        config_path = config_path.strip()
        if not config_path:
            continue
        config_path = os.path.abspath(config_path)
        tf.logging.info("Loading config from %s", config_path)
        with gfile.GFile(config_path.strip()) as config_file:
            config_flags = yaml.load(config_file)
            final_config = _deep_merge_dict(final_config, config_flags)

    # Merge flags with config values
    for flag_key, flag_value in final_config.items():
        if hasattr(flags, flag_key) and isinstance(getattr(flags, flag_key), dict):
            merged_value = _deep_merge_dict(flag_value, getattr(flags, flag_key))
            setattr(flags, flag_key, merged_value)
        elif hasattr(flags, flag_key):
            setattr(flags, flag_key, flag_value)
        else:
            tf.logging.warning("Ignoring config flag: %s", flag_key)
