import errno
import os


def create_directory(path):
    """Creates a local directory."""
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
