# -*- coding: utf-8 -*-
# @Time    : 2020/4/16 15:35
# @Author  : JiangYong Yu
import tensorflow as tf
from tensorflow.contrib import slim


def basic_block(x, planes, stride=1, downsample=None):
    """
    resnet18, 34 的残差block
    :param x:
    :param planes:
    :param stride:
    :param downsample:
    :return:
    """
    residual = x
    out = slim.conv2d(x, planes, 3, stride=stride)

    out = slim.conv2d(out, planes, 3, stride=1, activation_fn=None)
    if downsample is not None:
        residual = slim.conv2d(x, planes, 1, stride=stride, activation_fn=None)

    out += residual
    out = tf.nn.relu(out)
    return out


def make_layer(x, block, in_planes, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or in_planes != planes:
        downsample = True
    x = block(x, planes, stride=stride, downsample=downsample)
    in_planes = planes
    for i in range(1, blocks):
        x = block(x, planes, stride=1)
    return x, in_planes


def resnet_arg_scope(weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):

    """
    slim.conv2d的部分参数预定义，包含bn的参数，padding的参数
    :param weight_decay:
    :param batch_norm_decay:
    :param batch_norm_epsilon:
    :param batch_norm_scale:
    :return:
    """

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
    """
    resnet结构
    :param inputs: 输入图片，shape (batch, height, width, 3)
    :param block: basic_block
    :param layers: res18 [2, 2, 2, 2], res34 [3, 4, 6, 3]
    :param is_training: 训练用true， 测试用false
    :return:
    """
    with slim.arg_scope(resnet_arg_scope()) as sc:
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            x = slim.conv2d(inputs, 64, 7, stride=2)
            x = slim.max_pool2d(x, kernel_size=3, stride=2, padding='SAME')
            x1, in_planes = make_layer(x, block, 64, 64, layers[0])
            x2, in_planes = make_layer(x1, block, in_planes, 128, layers[1], stride=2)
            x3, in_planes = make_layer(x2, block, in_planes, 256, layers[2], stride=2)
            x4, _ = make_layer(x3, block, in_planes, 512, layers[2], stride=2)
    return x1, x2, x3, x4




