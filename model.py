import tensorflow as tf
from tensorflow.python.ops import array_ops
import math
from tensorflow.contrib import slim
import numpy as np
import resnet_18_34
import resnet_50
import config as cfg


def mean_image_subtraction(images):
    """
    输入图片 ÷ 255,减去均值，除以方差
    :param images:
    :return:
    """
    num_channels = images.get_shape().as_list()[-1]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    if len(mean) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    images = images / 255.0
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] = (channels[i] - mean[i]) / std[i]
    return tf.concat(axis=3, values=channels)


def model(images, weight_decay=1e-5, res_type="res18", is_training=True):
    """
    定义模型
    :param images:
    :param weight_decay:
    :param res_type:
    :param is_training:
    :return:
    """
    # 输入图片做normalize
    images = mean_image_subtraction(images)

    # 选取合适的主干网络
    if res_type == "res18":
        end_points = resnet_18_34.resnet(images, resnet_18_34.basic_block, [2, 2, 2, 2], is_training=is_training)
    elif res_type == "res34":
        end_points = resnet_18_34.resnet(images, resnet_18_34.basic_block, [3, 4, 6, 3], is_training=is_training)
    else:
        end_points = resnet_50.resnet(images, resnet_50.bottle_neck, [3, 4, 6, 3], is_training=is_training)

    # 取出FPN需要的层
    feature_dict = {'C2': end_points[0],
                    'C3': end_points[1],
                    'C4': end_points[2],
                    'C5': end_points[3]}

    # FPN操作
    pyramid_dict = {}
    with tf.variable_scope('build_pyramid'):
        with slim.arg_scope([slim.conv2d],
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            activation_fn=None,
                            normalizer_fn=None):
            last_fm = None
            for i in range(3):
                fm = feature_dict['C{}'.format(5 - i)]
                fm_1x1_conv = slim.conv2d(fm, num_outputs=cfg.fpn_out_channels, kernel_size=[1, 1],
                                          stride=1, scope='p{}_1x1_conv'.format(5 - i))
                if last_fm is not None:
                    h, w = tf.shape(fm_1x1_conv)[1], tf.shape(fm_1x1_conv)[2]
                    last_resize = tf.image.resize_bilinear(last_fm,
                                                           size=[h, w],
                                                           name='p{}_up2x'.format(5 - i))

                    fm_1x1_conv = fm_1x1_conv + last_resize

                last_fm = fm_1x1_conv

                fm_3x3_conv = slim.conv2d(fm_1x1_conv,
                                          num_outputs=cfg.fpn_out_channels, kernel_size=[3, 3], padding="SAME",
                                          stride=1, scope='p{}_3x3_conv'.format(5 - i))
                pyramid_dict['P{}'.format(5 - i)] = fm_3x3_conv

            p6 = slim.conv2d(pyramid_dict['P5'],
                             num_outputs=cfg.fpn_out_channels, kernel_size=[3, 3], padding="SAME",
                             stride=2, scope='p6_conv')
            pyramid_dict['P6'] = p6

            p7 = tf.nn.relu(p6)

            p7 = slim.conv2d(p7,
                             num_outputs=cfg.fpn_out_channels, kernel_size=[3, 3], padding="SAME",
                             stride=2, scope='p7_conv')

            pyramid_dict['P7'] = p7

    # FPN输出P3, P4, P5, P6, P7，分别是原图的1/8, 1/16, 1/32, 1/64, 1/128

    # head部分，这里没有用bn，原文作者使用的gn，所以我也使用的gn
    with tf.variable_scope('fcos_head'):
        # batch_norm_params = {
        #     'decay': 0.997,
        #     'epsilon': 1e-5,
        #     'scale': True,
        #     'is_training': is_training
        # }

        # gn的参数
        gn_norm_params = {
            'epsilon': 1e-5,
            'scale': True,
            'trainable': is_training
        }

        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.group_norm,
                            normalizer_params=gn_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):

            fpn_feature = [pyramid_dict['P3'], pyramid_dict['P4'], pyramid_dict['P5'],
                           pyramid_dict['P6'], pyramid_dict['P7']]

            logits = []
            bbox_reg = []
            mask_reg = []
            centerness_list = []

            for i, feature in enumerate(fpn_feature):
                cls_tower = feature
                for id_f in range(1):
                    cls_tower = slim.conv2d(cls_tower, cfg.fpn_out_channels, 3)

                bbox_tower = feature
                for bd_f in range(1):
                    bbox_tower = slim.conv2d(bbox_tower, cfg.fpn_out_channels, 3)

                mask_tower = feature
                for bd_f in range(1):
                    mask_tower = slim.conv2d(mask_tower, cfg.fpn_out_channels, 3)

                cls_logits = slim.conv2d(cls_tower, cfg.num_classes, 3, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                if is_training:
                    centerness = slim.conv2d(mask_tower, 1, 3, activation_fn=None, normalizer_fn=None)
                else:
                    centerness = slim.conv2d(mask_tower, 1, 3, activation_fn=tf.nn.sigmoid, normalizer_fn=None)

                bbox_pred = slim.conv2d(bbox_tower, 4, 3, activation_fn=tf.exp, normalizer_fn=None)
                mask_pred = slim.conv2d(mask_tower, 36, 3, activation_fn=tf.exp, normalizer_fn=None)

                logits.append(cls_logits)
                centerness_list.append(centerness)
                bbox_reg.append(bbox_pred)
                mask_reg.append(mask_pred)
    return logits, bbox_reg, mask_reg, centerness_list





