"""
Transforms that can be used with the data loader
"""
from collections import namedtuple

import skimage.io as skio
import skimage.transform as skt
import numpy as np
import skimage.color as skc


TransformFn = namedtuple('TransformFn', 'fn args')


class ImageTransform(object):
    def __init__(self):
        self.transform_fns = []

    def load(self):
        self.transform_fns.append(TransformFn(im_load, []))
        return self

    def resize(self, img_size):
        self.transform_fns.append(TransformFn(im_resize, [img_size]))
        return self

    def resample(self, rate):
        self.transform_fns.append(TransformFn(im_resample, [rate]))
        return self

    def resize_and_pad(self, img_size):
        self.transform_fns.append(TransformFn(im_resize_and_pad, [img_size]))
        return self

    def crop_and_resize(self, img_size):
        self.transform_fns.append(TransformFn(im_crop_and_resize, [img_size]))
        return self

    def align(self, step):
        self.transform_fns.append(TransformFn(im_align, [step]))
        return self

    def from_label(self, img_size):
        self.transform_fns.append(TransformFn(im_from_label, [img_size]))

    def build(self):
        def wrapper(im):
            if isinstance(im, str):
                im = im_load(im)
            im = im_from_uint8(im)
            for transform_fn in self.transform_fns:
                im = transform_fn[0](im, *transform_fn[1])
            im = im_to_uint8(im)
            im = im_expand_channel(im)
            return im
        return wrapper


def im_resize_and_pad(im, img_size):
    # Get the ratio of width to height for each
    current_whratio = im.shape[1] / im.shape[0]
    desired_whratio = img_size[1] / img_size[0]
    # Check if the image has roughly the same ratio, else pad it
    if np.round(im.shape[0] * desired_whratio) != im.shape[1]:
        height = im.shape[0]
        width = im.shape[1]
        # Desired shape is wider than current one
        if desired_whratio > current_whratio:
            half = np.round(height * desired_whratio)
            height_pad_start = 0
            height_pad_end = 0
            width_pad_start = int(abs(np.floor((width - half) / 2)))
            width_pad_end = int(abs(np.ceil((width - half) / 2)))
        # Desired shape is taller than current
        else:
            half = np.round(width / desired_whratio)
            height_pad_start = int(abs(np.floor((height - half) / 2)))
            height_pad_end = int(abs(np.ceil((height - half) / 2)))
            width_pad_start = 0
            width_pad_end = 0
        # Constant value to pad with
        consts = [np.median(np.concatenate((im[0, :, i], im[-1, :, i], im[:, 0, i], im[:, -1, i]))) for i in
                  range(im.shape[2])]
        # Pad
        im = np.stack(
            [np.pad(im[:, :, c],
                    ((height_pad_start, height_pad_end), (width_pad_start, width_pad_end)),
                    mode='constant',
                    constant_values=consts[c])
             for c in range(im.shape[2])], axis=2)
    # Resize
    if im.dtype == np.uint8:
        im = im / 255
    if im.shape[0] != img_size[0] or im.shape[1] != img_size[1]:
        im = skt.resize(im, img_size)
    return im


def im_crop_and_resize(im, img_size):
    # Get the ratio of width to height for each
    current_whratio = im.shape[1] / im.shape[0]
    desired_whratio = img_size[1] / img_size[0]
    # Check if the image has roughly the same ratio, else crop it
    if np.round(im.shape[0] * desired_whratio) != im.shape[1]:
        height = im.shape[0]
        width = im.shape[1]
        # Desired shape is taller than current one
        if desired_whratio < current_whratio:
            half = np.round(height * desired_whratio)
            height_offset_start = 0
            height_offset_end = height
            width_offset_start = int(abs(np.floor((width - half) / 2)))
            width_offset_end = width - int(abs(np.ceil((width - half) / 2)))
        # Desired shape is wider than current
        else:
            half = np.round(width / desired_whratio)
            height_offset_start = int(abs(np.floor((height - half) / 2)))
            height_offset_end = height - int(abs(np.ceil((height - half) / 2)))
            width_offset_start = 0
            width_offset_end = width
        # Crop
        im = im[height_offset_start:height_offset_end, width_offset_start:width_offset_end, ...]
    # Resize
    if im.dtype == np.uint8:
        im = im / 255
    if im.shape[0] != img_size[0] or im.shape[1] != img_size[1]:
        im = skt.resize(im, img_size)
    return im


def im_resize(im, img_size):
    if im.shape[0] != img_size[0] or im.shape[1] != img_size[1]:
        im = skt.resize(im, img_size)
    return im


def im_resample(im, rate):
    if rate is not None and rate > 1:
        return im[::rate, ::rate, ...]
    else:
        return im


def im_align(im, step):
    if step is not None and step > 1:
        h = (im.shape[0] // step) * step
        w = (im.shape[1] // step) * step
        return im[:h, :w, ...]
    else:
        return im


def im_load(filename):
    return skio.imread(filename)


def im_to_greyscale(im, flag=True):
    if flag and np.ndim(im) == 3 and im.shape[-1] != 1:
        im = skc.rgb2gray(im)
    return im


def im_from_uint8(im):
    if im.dtype == np.uint8:
        im = im / 255
    return im


def im_to_uint8(im):
    if im.dtype == np.float:
        im = (im * 255).astype(np.uint8)
    return im


def im_expand_channel(im):
    if np.ndim(im) == 2:
        im = im[..., np.newaxis]
    return im


def im_from_label(lab, img_size):
    return np.ones(img_size) * lab


def preprocess(resize_shape=None,
               align_step=None,
               resample_rate=None,
               pad_to_resize_shape=True,
               to_greyscale=False,
               is_filename=True,
               as_float=False,
               expand_single_channel=True):
    def wrapper(im):
        if is_filename:
            im = skio.imread(im)
        if to_greyscale and np.ndim(im) == 3 and im.shape[-1] != 1:
            im = skc.rgb2gray(im)
        if im.dtype == np.uint8:
            im = im / 255
        if resize_shape is not None:
            if pad_to_resize_shape:
                im = im_resize_and_pad(im, resize_shape)
            else:
                im = im_resize(im, resize_shape)
        if resample_rate is not None:
            im = im_resample(im, resample_rate)
        if align_step is not None:
            im = im_align(im, align_step)
        if not as_float:
            im = (im * 255).astype(np.uint8)
        if expand_single_channel and np.ndim(im) == 2:
            im = im[..., np.newaxis]
        if im.shape[-1] > 3:
            im = im[..., 0:3]
        return im
    return wrapper


def annotations_to_label_image(idx, annotations, labels, shape, reduce_factor=1):
    im = np.zeros(shape)
    for ann in annotations:
        if ann.label in labels:
            i = labels.index(ann.label)
            xs = ann.x // reduce_factor
            xe = (ann.x + ann.width) // reduce_factor
            ys = ann.y // reduce_factor
            ye = (ann.y + ann.height) // reduce_factor
            im[ys:ye, xs:xe, ...] = i + 1
    return im


class LabelTransform(object):
    def __init__(self):
        self.transform_fns = []

    def from_label(self, img_size):
        self.transform_fns.append(TransformFn(im_from_label, [img_size]))
        return self

    def build(self):
        def wrapper(lab):
            for transform_fn in self.transform_fns:
                lab = transform_fn[0](lab, *transform_fn[1])
            return lab
        return wrapper

#
#
#
# def rescale_transform(idx, filename, factor):
#     im = skio.imread(filename)
#     if np.ndim(im) <= 3:
#         im = skt.rescale(im, [factor, factor, 1])
#     else:
#         im = skt.rescale(im, [1, factor, factor, 1])
#     return (im * 255).astype(np.uint8)
#
#
# def resample_transform(idx, filename, step):
#     im = skio.imread(filename)
#     return im[::step, ::step, ...]
#
#
# def resample_and_align_transform(idx, filename, step, align):
#     im = skio.imread(filename)
#     h = (im.shape[0] // (step * align)) * (step * align)
#     w = (im.shape[1] // (step * align)) * (step * align)
#     return im[:h:step, :w:step, ...]
#
#
# def resize_transform(idx, filename, shape):
#     im = skio.imread(filename)
#     if np.ndim(im) <= 3:
#         if im.shape[0] == shape[0] and im.shape[1] == shape[1]:
#             return im
#         im = skt.resize(im, shape)
#     else:
#         if im.shape[1] == shape[0] and im.shape[2] == shape[1]:
#             return im
#         im = skt.resize(im, [1, *shape])
#     return (im * 255).astype(np.uint8)
#
#
# def resize_with_pad_transform(idx, filename, shape):
#     im = skio.imread(filename)
#     if np.ndim(im) <= 3:
#         if im.shape[0] == shape[0] and im.shape[1] == shape[1]:
#             return im
#         im = resize_and_pad_image(im, shape)
#     else:
#         if im.shape[1] == shape[0] and im.shape[2] == shape[1]:
#             return im
#         im = resize_and_pad_image(im[0], shape)[np.newaxis, ...]
#     return (im * 255).astype(np.uint8)
#
#
# def load_image(idx, val, args):
#     return skio.imread(val)
#
#
# def align(idx, val, step):
#     im = skio.imread(val)
#     h = (im.shape[0] // step) * step
#     w = (im.shape[1] // step) * step
#     return im[:h, :w, ...]
#
#
# def null_transform(idx, val, args):
#     return val


def annotations_to_label_image(idx, annotations, labels, shape, reduce_factor=1):
    im = np.zeros(shape)
    for ann in annotations:
        if ann.label in labels:
            i = labels.index(ann.label)
            xs = ann.x // reduce_factor
            xe = (ann.x + ann.width) // reduce_factor
            ys = ann.y // reduce_factor
            ye = (ann.y + ann.height) // reduce_factor
            im[ys:ye, xs:xe, ...] = i + 1
    return im
