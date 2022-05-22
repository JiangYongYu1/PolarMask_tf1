import tensorflow as tf
import config as cfg
from tensorflow.python.ops import array_ops


def focal_loss(pred, label, background=0, alpha=0.5, gamma=2.0):
    # focal loss的tf实现
    label = tf.cast(label[:, :, 0], tf.int32)

    one_hot = tf.one_hot(label, cfg.num_classes + 1, axis=2)

    # one_hot = tf.squeeze(one_hot, axis=-1)

    onehot = one_hot[:, :, 1:]

    pred = tf.clip_by_value(pred, 1e-6, 1 - 1e-6)

    pos_part = tf.pow(1 - pred, gamma) * onehot * tf.log(pred)
    neg_part = tf.pow(pred, gamma) * (1 - onehot) * tf.log(1 - pred)
    loss = tf.reduce_sum(-(alpha * pos_part + (1 - alpha) * neg_part), axis=2)
    positive_mask = tf.cast(tf.greater(label, background), tf.float32)
    return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(positive_mask), 1)


# def focal__(y_pred, label, background=0, alpha=0.5, gamma=2.0):
#     label = tf.cast(label[:, :, 0], tf.int32)
#     one_hot = tf.one_hot(label, cfg.num_classes + 1, axis=2)
#     y_true = one_hot[:, :, 1:]
#
#     y_true = tf.cast(y_true, tf.float32)
#     y_pred = tf.clip_by_value(y_pred, cfg.epsilon, 1. - cfg.epsilon)
#
#     alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
#     y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
#     ce = -tf.log(y_t)
#     weight = tf.pow(tf.subtract(1., y_t), gamma)
#     fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
#     loss = tf.reduce_mean(fl)
#     return loss
#
#
# def focal_(pred, label, background=0, alpha=0.5, gamma=2.0):
#     label = tf.cast(label, tf.int32)
#     one_hot = tf.one_hot(label, cfg.num_classes + 1, axis=2)
#     one_hot = tf.squeeze(one_hot, axis=-1)
#     labels = one_hot[:, :, 1:]
#
#     alpha_factor = tf.ones_like(labels) * alpha
#     alpha_factor = tf.where(tf.equal(labels, 1), alpha_factor, 1 - alpha_factor)
#     # (1 - 0.99) ** 2 = 1e-4, (1 - 0.9) ** 2 = 1e-2
#     focal_weight = tf.where(tf.equal(labels, 1), 1 - pred, pred)
#     focal_weight = alpha_factor * focal_weight ** gamma
#     cls_loss = focal_weight * tf.keras.backend.binary_crossentropy(labels, pred)
#
#     positive_mask = tf.cast(tf.greater(label, background), tf.float32)
#     return tf.reduce_sum(cls_loss) / tf.maximum(tf.reduce_sum(positive_mask), 1)


def iou_loss(pred, target, weight):
    """
    box iou loss
    :param pred:
    :param target:
    :param weight:
    :return:
    """
    pred = tf.abs(pred)
    target = tf.abs(target)

    g_a, g_b, g_c, g_d = tf.split(target, 4, axis=1)
    p_a, p_b, p_c, p_d = tf.split(pred, 4, axis=1)

    gt_area = (g_a + g_c) * (g_b + g_d)
    pred_area = (p_a + p_c) * (p_b + p_d)

    w_union = tf.minimum(g_a, p_a) + tf.minimum(g_c, p_c)
    h_union = tf.minimum(g_b, p_b) + tf.minimum(g_d, p_d)

    area_intersect = w_union * h_union
    area_union = gt_area + pred_area - area_intersect

    o_loss = - tf.log((area_intersect + 1) / (area_union + 1))
    o_r_loss = tf.reduce_sum(o_loss * weight) / tf.maximum(tf.reduce_sum(weight), 1)
    return o_r_loss


def g_iou_loss(pred, target, centerness_targets):
    g_a, g_b, g_c, g_d = tf.split(target, 4, axis=2)
    p_a, p_b, p_c, p_d = tf.split(pred, 4, axis=2)

    gt_area = (g_a + g_c) * (g_b + g_d)
    pred_area = (p_a + p_c) * (p_b + p_d)

    w_union = tf.minimum(g_a, p_a) + tf.minimum(g_c, p_c)
    h_union = tf.minimum(g_b, p_b) + tf.minimum(g_d, p_d)

    g_w_intersect = tf.maximum(g_a, p_a) + tf.maximum(g_c, p_c)
    g_h_intersect = tf.maximum(g_b, p_b) + tf.maximum(g_d, p_d)
    ac_uion = g_w_intersect * g_h_intersect + 1e-7

    area_intersect = w_union * h_union
    area_union = gt_area + pred_area - area_intersect

    ious = (area_intersect + 1) / (area_union + 1)
    gious = ious - (ac_uion - area_union) / ac_uion

    losses = 1 - gious
    losses = tf.reduce_sum(losses * centerness_targets) / tf.maximum(tf.reduce_sum(centerness_targets), 1.0)
    return losses


def center_loss(pred, target):
    center_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=target)
    centerness_loss = tf.reduce_mean(center_loss)
    return centerness_loss


def tensor_shape(x, rank=3):
    """Returns the dimensions of a tensor.
    Args:
      image: A N-D Tensor of shape.
    Returns:
      A list of dimensions. Dimensions that are statically known are python
        integers,otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


def compute_mask_centerness_targets_pos(reg_mask_targets):
    """

    :param reg_mask_targets:
    :return:
    """
    l_min = tf.reduce_min(reg_mask_targets, axis=-1)
    l_max = tf.reduce_max(reg_mask_targets, axis=-1)

    centerness = tf.sqrt(l_min / l_max)
    centerness = tf.expand_dims(centerness, axis=-1)
    return centerness


def compute_mask_centerness_targets(reg_mask_targets, pos_input_mask_masks):
    """

    :param reg_mask_targets:
    :param pos_input_mask_masks:
    :return:
    """
    neg_mask = 1 - pos_input_mask_masks
    neg_big_mask = neg_mask * cfg.INF
    neg_small_mask = -neg_mask * cfg.INF

    l_min = tf.reduce_min(reg_mask_targets + neg_big_mask, axis=-1)
    l_max = tf.reduce_max(reg_mask_targets + neg_small_mask, axis=-1)

    zeros = array_ops.zeros_like(l_min, dtype=l_min.dtype)

    centerness = tf.where(tf.equal(l_max, 0.0), zeros,
                          tf.sqrt(l_min / l_max))
    centerness = tf.expand_dims(centerness, axis=-1)
    return centerness


def compute_centerness_targets(reg_targets_a):
    reg_targets = tf.abs(reg_targets_a)

    delta_left = reg_targets[..., 0]
    delta_right = reg_targets[..., 2]
    delta_top = reg_targets[..., 1]
    delta_down = reg_targets[..., 3]

    l13 = tf.stack([delta_left, delta_right], axis=-1)
    l24 = tf.stack([delta_top, delta_down], axis=-1)

    max_l13 = tf.reduce_max(l13, axis=-1)
    min_l13 = tf.reduce_min(l13, axis=-1)

    max_l24 = tf.reduce_max(l24, axis=-1)
    min_l24 = tf.reduce_min(l24, axis=-1)

    zeros = array_ops.zeros_like(min_l13, dtype=min_l13.dtype)

    centerness = tf.where(tf.logical_or(tf.equal(max_l13, 0.0),  tf.equal(max_l24, 0.0)), zeros,
                          tf.sqrt(min_l13 / max_l13 * min_l24 / max_l24))
    centerness = tf.expand_dims(centerness, axis=-1)
    return centerness


def mask_iou_loss(pred, target, weight, pos_input_mask_masks):
    """
    mask iou loss
    :param pred:
    :param target:
    :param weight:
    :param pos_input_mask_masks:
    :return:
    """
    total = tf.stack([pred, target], axis=-1)
    l_max = tf.reduce_max(total, axis=-1) * pos_input_mask_masks
    l_min = tf.reduce_min(total, axis=-1) * pos_input_mask_masks

    # loss = - tf.log((tf.reduce_sum(l_min, axis=1) + 1) / (tf.reduce_sum(l_max, axis=1) + 1))
    loss = tf.log((tf.reduce_sum(l_max, axis=1) + 1) / (tf.reduce_sum(l_min, axis=1) + 1))
    loss = tf.expand_dims(loss, axis=-1) * weight
    loss = tf.reduce_sum(loss) / (tf.reduce_sum(weight) + 1e-5)
    return loss


def total_loss(cls_pred, target_pred, center_pred, mask_target_pred, input_labels, input_targets,
               input_mask_targets, input_mask_masks, input_mask_centers, background=0):
    # 模型的输出和真值一起计算loss，loss包括四个部分，分类loss，中心区域loss, 框回归的loss，36个点的loss
    """

    :param cls_pred:
    :param target_pred:
    :param center_pred:
    :param mask_target_pred:
    :param input_labels:
    :param input_targets:
    :param input_mask_targets:
    :param input_mask_masks:
    :param input_mask_centers:
    :param background:
    :return:
    """
    box_cls_flatten = []
    box_regression_flatten = []
    centerness_flatten = []
    mask_regression_flatten = []

    # 第一步：模型输出是四个list, 分别对应cls, center, box, mask, 每个list里面有多个层的输出，需要全部拉直，再concat，
    # 因为输入的gt_cls, gt_center, gt_box, gt_mask都是这么做的，需要对应起来
    for tt_i in range(len(cfg.fpn_strides)):
        _, H, W, _ = tensor_shape(cls_pred[tt_i], 4)
        box_cls_flatten.append(tf.reshape(cls_pred[tt_i], (-1, H*W, cfg.num_classes)))
        box_regression_flatten.append(tf.reshape(target_pred[tt_i], (-1, H*W, 4)))
        centerness_flatten.append(tf.reshape(center_pred[tt_i], (-1, H*W, 1)))
        mask_regression_flatten.append(tf.reshape(mask_target_pred[tt_i], (-1, H*W, 36)))

    box_cls_flatten = tf.concat(box_cls_flatten, axis=1)
    box_regression_flatten = tf.concat(box_regression_flatten, axis=1)
    mask_regression_flatten = tf.concat(mask_regression_flatten, axis=1)
    centerness_flatten = tf.concat(centerness_flatten, axis=1)

    reg_targets_flatten = input_targets
    labels_flatten = tf.expand_dims(input_labels, axis=-1)

    # 计算分类loss，采用focal loss
    class_loss = focal_loss(box_cls_flatten, labels_flatten)

    # 这一步是判断输入是不是存在正样本，得到正样本的indices
    mask = 1 - tf.cast(tf.equal(labels_flatten[:, :, 0], background), tf.float32)
    indices = tf.where(tf.equal(mask, 1))

    # 如果全是负样本，centerness, box和mask的loss计算就没有意义了，直接输出0.0
    if tf.size(indices) == 0:
        reg_loss = tf.constant(0.0)
        centerness_loss = tf.constant(0.0)
        mask_loss = tf.constant(0.0)
    else:
        # 得到所有的正样本的点的预测值
        pos_box_regression_flatten = tf.gather_nd(box_regression_flatten, indices)
        pos_reg_targets_flatten = tf.gather_nd(reg_targets_flatten, indices)
        pos_input_mask_targets = tf.gather_nd(input_mask_targets, indices)
        pos_input_mask_masks = tf.gather_nd(input_mask_masks, indices)

        center_map_flatten = tf.gather_nd(input_mask_centers, indices)

        pos_centerness_flatten = tf.gather_nd(centerness_flatten, indices)

        # 计算回归loss，这里直接用的iou loss，因为可能出现真实和预测不重叠的情况，所以iou loss 和 giou loss没有区别
        reg_loss = iou_loss(pos_box_regression_flatten, pos_reg_targets_flatten, center_map_flatten)

        # 计算中心区域loss，这里采用的是交叉熵loss
        centerness_loss = center_loss(pos_centerness_flatten, center_map_flatten)

        pos_mask_regression_flatten = tf.gather_nd(mask_regression_flatten, indices)
        # pos_input_mask_targets = tf.gather_nd(input_mask_targets, indices)

        # 计算mask 回归loss
        mask_loss = mask_iou_loss(pos_mask_regression_flatten, pos_input_mask_targets, center_map_flatten,
                                  pos_input_mask_masks)

    tf.summary.scalar("class_loss", class_loss)
    tf.summary.scalar("reg_loss", reg_loss)
    tf.summary.scalar("centerness_loss", centerness_loss)
    tf.summary.scalar("mask_loss", mask_loss)

    return class_loss, reg_loss, centerness_loss, mask_loss
