# -*- coding: utf-8 -*-
# @Time    : 2020/4/28 19:45
# @Author  : JiangYong Yu
import tensorflow as tf
from tensorflow.contrib import slim


def bottle_neck(x, planes, stride=1, downsample=None):
    residual = x
    out = slim.conv2d(x, planes, 1, stride=1)

    out = slim.conv2d(out, planes, 3, stride=stride)
    out = slim.conv2d(out, planes*4, 1, activation_fn=None)
    if downsample is not None:
        residual = slim.conv2d(x, planes*4, 1, stride=stride, activation_fn=None)

    out += residual
    out = tf.nn.relu(out)
    return out


def make_layer(x, block, in_planes, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or in_planes != planes * 4:
        downsample = True
    x = block(x, planes, stride=stride, downsample=downsample)
    in_planes = planes * 4
    for i in range(1, blocks):
        x = block(x, planes, stride=1)
    return x, in_planes


def resnet_arg_scope(weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc


def resnet(inputs, block, layers, is_training=True):
    with slim.arg_scope(resnet_arg_scope()) as sc:
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            x = slim.conv2d(inputs, 64, 7, stride=2)
            x = slim.max_pool2d(x, kernel_size=3, stride=2, padding='SAME')
            x1, in_planes = make_layer(x, block, 64, 64, layers[0])
            x2, in_planes = make_layer(x1, block, in_planes, 128, layers[1], stride=2)
            x3, in_planes = make_layer(x2, block, in_planes, 256, layers[2], stride=2)
            x4, _ = make_layer(x3, block, in_planes, 512, layers[3], stride=2)
    return x1, x2, x3, x4




