import tensorflow as tf

from utils import function_wrapper


# -------------------------------------------------------------------------------------------
# RandomCrop
# -------------------------------------------------------------------------------------------
def random_crop_image(im, size):
    image_shape = tf.shape(im)
    im = tf.image.pad_to_bounding_box(
        im,
        0,
        0,
        tf.maximum(size[0], image_shape[0]),
        tf.maximum(size[1], image_shape[1]))
    im = tf.image.random_crop(im, size=tf.concat([size, [image_shape[-1]]], axis=0))
    return im


def random_crop_image_label(im, label, size):
    combined = tf.concat([im, tf.expand_dims(label, 2)], axis=2)
    image_shape = tf.shape(im)
    combined_crop = random_crop_image(combined, size)
    return (combined_crop[:, :, :image_shape[-1]],
            combined_crop[:, :, image_shape[-1]])


def random_crop_image_label_mask(im, label, mask, size):
    combined = tf.concat([im, tf.expand_dims(label, 2), tf.expand_dims(mask, 2)], axis=2)
    image_shape = tf.shape(im)
    combined_crop = random_crop_image(combined, size)
    return (combined_crop[:, :, :image_shape[-1]],
            combined_crop[:, :, image_shape[-1]],
            combined_crop[:, :, image_shape[-1]+1])


def RandomCrop(num_inputs, size):
    if num_inputs == 1:
        return function_wrapper(random_crop_image, 1, [size])
    elif num_inputs == 2:
        return function_wrapper(random_crop_image_label, 2, [size])
    elif num_inputs == 3:
        return function_wrapper(random_crop_image_label_mask, 3, [size])
    else:
        raise ValueError("Number of inputs must be 1 - 3")


# -------------------------------------------------------------------------------------------
# Identity
# -------------------------------------------------------------------------------------------
def identity(t):
    return t


def Identity():
    return function_wrapper(identity, 1, [])


# -------------------------------------------------------------------------------------------
# Divide
# -------------------------------------------------------------------------------------------
def divide(t, val):
    t = tf.cast(t, tf.float32)
    return tf.divide(t, val)


def Divide(val):
    return function_wrapper(divide, 1, [val])

