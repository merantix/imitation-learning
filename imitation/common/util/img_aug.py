import tensorflow as tf


def get_hwc(img):
    shape = img.shape.as_list()
    assert len(shape) == 3, 'Expected (H, W, C). Found {}'.format(shape)
    return shape


def _add_batch_dim(img):
    """Many TF functions require NWHC input.  Convert WHC image to NWHC of batch size 1."""
    get_hwc(img)  # validate dimensions
    return tf.expand_dims(img, 0)


def _remove_batch_dim(img_batch):
    """Many TF functions require NWHC format. Convert NHWC image (of batch size 1) to HWC image."""
    shape = img_batch.shape.as_list()
    assert len(shape) == 4, 'Expected (N, W, H, C). Found {}'.format(shape)
    assert shape[0] == 1, 'Expected singleton batch.  Found {} images'.format(shape[0])
    return tf.squeeze(img_batch, axis=0)


def gaussian_kernel(mean=0.0, stddev=1.0):
    """Make 2D gaussian Kernel for convolution."""
    d = tf.distributions.Normal(mean, stddev)
    size = tf.math.ceil(2 * stddev)
    vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))

    # combine two 1D gaussian kernels into one 2D kernel
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    normalized_kernel = gauss_kernel / tf.reduce_sum(gauss_kernel)
    return normalized_kernel


def gauss_blur(img, mean=0.0, stddev=1.0):
    """Apply guassian blur to an image w/ given sigma."""
    _, _, num_channels = get_hwc(img)
    gauss_kernel = gaussian_kernel(mean=mean, stddev=stddev)
    gauss_kernel = tf.stack([gauss_kernel] * num_channels, axis=2)
    gauss_kernel = gauss_kernel[:, :, :, tf.newaxis]
    img_batch = _add_batch_dim(img)  # conv2d expects NWHC
    blurred_batch = tf.nn.depthwise_conv2d(img_batch, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
    blurred = _remove_batch_dim(blurred_batch)
    return blurred


def gauss_noise(img, per_channel, mean=0.0, stddev=1.0):
    """Applies additive gauss noise to an image with the given mean and stddev.

    Args:
        img: a [H, W, C] image
        per_channel: (bool) apply separate noise to each channel, or the same noise to all channels?
        mean: (float) mean of gauss distribution
        stddev: (float) stddev of gauss distribution

    Returns:
        Perturbed image
    """
    h, w, c = get_hwc(img)
    if per_channel:
        noise = tf.random_normal((h, w, c), mean=mean, stddev=stddev, dtype=tf.float32)
    else:
        channel_noise = tf.random_normal((h, w), mean=mean, stddev=stddev, dtype=tf.float32)
        noise = tf.stack([channel_noise] * c, axis=2)
    new_img = img + noise
    return new_img


def _get_p_channel_drop(p_pixel_drop):
    """Given a desired probability a pixel is dropped, computes the probability a channel is dropped for RGB image.

    P(pixel_dropped)
    = 1 - P(pixel_not_dropped)
    = 1 - P(channel_not_dropped)^num_channels
    = 1 - (1 - p_channel_drop)^3
    => p_channel_drop = 1 - cube_root(1 - p_pixel_drop)
    """
    p_channel_drop = 1 - tf.math.pow(1 - p_pixel_drop, 1 / 3.)
    p_channel_drop = tf.minimum(tf.maximum(0., p_channel_drop), 1.)
    return p_channel_drop


def _build_dropout_mask(height, width, p_pixel_drop, per_channel):
    """Build dropout mask for multiplying with an image.

    Args:
        height: height of mask in pixels
        width: width of mask in pixels
        p_pixel_drop: probability that a given pixel is dropped
        per_channel: if True, drop individual channels

    Returns:
        a 0-1 valued tensor of shape [height, width, 3]
    """
    if per_channel:
        p_channel_drop = _get_p_channel_drop(p_pixel_drop)
        dist = tf.distributions.Bernoulli(probs=(1 - p_channel_drop))
        mask = tf.cast(dist.sample((height, width, 3)), tf.float32)
    else:
        dist = tf.distributions.Bernoulli(probs=(1 - p_pixel_drop))
        channel_mask = tf.cast(dist.sample((height, width)), tf.float32)
        mask = tf.stack([channel_mask] * 3, axis=2)
    return mask


def pixelwise_dropout(img, p_pixel_drop, per_channel):
    """Randomly drop pixels.

    Args:
        img: HWC image
        p_pixel_drop: probability an individual pixel is dropped
        per_channel: if True, apply channel-by-channel.  if False, set dropped pixels to black

    Returns:
        HWC image with randomly dropped pixels
    """
    h, w, _ = get_hwc(img)
    mask = _build_dropout_mask(h, w, p_pixel_drop, per_channel)
    masked = tf.multiply(img, mask)
    return masked


def coarse_pixelwise_dropout(img, p_height, p_width, p_pixel_drop, per_channel):
    """Randomly drop rectangles of pixels.

    Args:
        img: HWC image
        p_height: height proportion at which dropout mask is generated
                (e.g. setting to 0.5 will result in dropped rectangles of with 2 pixel height)
        p_width: analagous to above
        p_pixel_drop: probability a given pixel is dropped
        per_channel: if True, apply channel-by-channel. if False, set dropped pixels to black.

    Returns:
        HWC image with randmoly dropped rectangles of pixels

    """
    h, w, c = get_hwc(img)
    assert c == 3, 'Expected RGB image. Found {} channels'.format(c)

    # Generate a 'downsampled' single-pixel dropout mask. Then upsample it so that each pixel becomes rectangular region
    # 1. get downsized mask
    mask_h = tf.cast(tf.round(h * p_height), dtype=tf.int32)
    mask_w = tf.cast(tf.round(w * p_width), dtype=tf.int32)
    small_mask = _build_dropout_mask(mask_h, mask_w, p_pixel_drop, per_channel)

    # 2. upsample mask
    small_mask_batch = _add_batch_dim(small_mask)
    mask_batch = tf.image.resize_nearest_neighbor(images=small_mask_batch, size=(h, w), align_corners=True)
    mask = _remove_batch_dim(mask_batch)

    masked = tf.multiply(img, mask)
    return masked
