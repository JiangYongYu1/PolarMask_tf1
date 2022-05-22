import time
import config as cfgs
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import cv2
import model
import data_generator_for_polar_mask as icdar
import loss_ad as loss


tf.app.flags.DEFINE_integer('input_size_w', cfgs.img_size_w, '')
tf.app.flags.DEFINE_integer('input_size_h', cfgs.img_size_h, '')
tf.app.flags.DEFINE_integer('batch_size_per_gpu', cfgs.batch_size, '')
tf.app.flags.DEFINE_integer('num_readers', 32, '')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, '')
tf.app.flags.DEFINE_integer('max_steps', 200000, '')
tf.app.flags.DEFINE_integer('warmup_steps', 5000, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.999, '')
tf.app.flags.DEFINE_string('gpu_list', '0,1', '')
tf.app.flags.DEFINE_string('checkpoint_path',
                           '/home/jyyu/models/cocoapi-master/model_files/polarmask/checkpoint_path_big/', '')
tf.app.flags.DEFINE_boolean('restore', True, 'whether to resotre from checkpoint')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 1000, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 50, '')
tf.app.flags.DEFINE_string('pretrained_model_path',
                           "/home/jyyu/models/cocoapi-master/model_files/polarmask/checkpoint_path_small/", '')

FLAGS = tf.app.flags.FLAGS

gpus = list(range(len(FLAGS.gpu_list.split(','))))


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


def draw_gt_tf(gt_score_map, in_img):
    locations = compute_locations(cfgs.feat_shape)

    def draw_gt_cv(score_geo_map, location):
        """

        :param score_geo_map:
        :param location:
        :return:
        """
        score = score_geo_map[:, 0]
        geo = score_geo_map[:, 1:5]
        mask_geo = score_geo_map[:, 5:]

        select_idx = score > 0

        s_score = score[select_idx]
        s_geo = geo[select_idx]
        s_locate = location[select_idx]
        s_mask_geo = mask_geo[select_idx]

        s_detection = np.stack([
            s_locate[:, 0] - s_geo[:, 0],
            s_locate[:, 1] - s_geo[:, 1],
            s_locate[:, 0] + s_geo[:, 2],
            s_locate[:, 1] + s_geo[:, 3]], axis=-1)

        angle_stride = np.arange(0, 360, 10)
        sin_cos = np.stack([np.cos(angle_stride*np.pi/180), np.sin(angle_stride*np.pi/180)], axis=-1)
        s_mask_delta = np.expand_dims(sin_cos, axis=0) * np.expand_dims(s_mask_geo, axis=2)

        s_mask_points = np.expand_dims(s_locate, axis=1) + s_mask_delta[..., ::-1]

        s_detection = s_detection.reshape((-1, 2, 2))
        return s_detection, s_score, s_mask_points

    def draw_gt_cv_all(gt_score_map, img):
        """

        :param gt_score_map
        :param img:
        :return:
        """
        num_points_per_level = [len(points_per_level) for points_per_level in locations]
        for num_i, nums_level in enumerate(num_points_per_level):
            if num_i == len(num_points_per_level) - 1:
                continue
            num_points_per_level[num_i + 1] += nums_level
        score_geo_map_list = np.split(gt_score_map, num_points_per_level, axis=0)[:-1]
        img = img.astype(np.uint8)
        draw_img_list = []
        draw_score_list = []
        draw_mask_list = []
        for f_i in range(len(score_geo_map_list)):
            p_box, p_score, p_mask_point = draw_gt_cv(score_geo_map_list[f_i], locations[f_i])
            draw_img_list.append(p_box)
            draw_score_list.append(p_score)
            draw_mask_list.append(p_mask_point)
        draw_img_list = np.concatenate(draw_img_list, axis=0)
        draw_score_list = np.concatenate(draw_score_list, axis=0)
        draw_mask_list = np.concatenate(draw_mask_list, axis=0)
        for box, so, mask_point in zip(draw_img_list, draw_score_list, draw_mask_list):
            draw_box = box.astype(np.int)
            name_c = cfgs.idx_names_dict[so]
            draw_mask_point = mask_point.astype(np.int)

            img = cv2.putText(img, name_c, (draw_box[0][0], draw_box[0][1]), cv2.FONT_HERSHEY_SIMPLEX,
                              0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            img = cv2.rectangle(img, (draw_box[0][0], draw_box[0][1]), (draw_box[1][0], draw_box[1][1]), (255, 0, 0), 2)

            img = cv2.drawKeypoints(img, cv2.KeyPoint_convert(draw_mask_point[:, np.newaxis, :]), None, color=(0, 255, 0))
            # img = cv2.rectangle(img, tuple(box[0].astype(np.int)), tuple(box[1].astype(np.int)), True, (255, 0, 0), 2)
        img = img.astype(np.float32)
        return img

    img_draw_list = tf.py_func(draw_gt_cv_all,
                               inp=[gt_score_map, in_img],
                               Tout=tf.float32)
    return img_draw_list


def compute_locations_per_level(h, w, stride):
    x_centers = (np.arange(w, dtype=np.float32) + 0.5) * stride
    y_centers = (np.arange(h, dtype=np.float32) + 0.5) * stride
    x_centers, y_centers = np.meshgrid(x_centers, y_centers)

    y_centers = np.reshape(y_centers, (-1))
    x_centers = np.reshape(x_centers, (-1))

    locations = np.stack((x_centers, y_centers), axis=-1)
    return locations


def compute_locations(features):
    locations = []
    for level, feature in enumerate(features):
        h, w = feature
        locations_per_level = compute_locations_per_level(
            h, w, cfgs.fpn_strides[level])
        locations.append(locations_per_level)
    return locations


def draw_pred_tf(cls_flatten, regression_flatten, centerness_flatten, in_img):
    locations = compute_locations(cfgs.feat_shape)

    def draw_gt_cv(cls_map, target_map, centerness_map, location):
        """

        :param score_geo_map:
        :param location:
        :return:
        """
        score = cls_map
        geo = target_map

        geo = geo.reshape((-1, 4))
        score = score.reshape((-1, cfgs.num_classes))
        cntness = centerness_map.reshape((-1, 1))

        score = np.max(score, axis=-1)

        select_idx = score > cfgs.pre_nms_thresh

        s_geo = geo[select_idx]
        s_locate = location[select_idx]
        s_score = score[select_idx]
        s_cntness = cntness[select_idx]

        sc_score = np.expand_dims(s_score, axis=-1) * s_cntness
        s_detection = np.stack([
            s_locate[:, 0] - s_geo[:, 0],
            s_locate[:, 1] - s_geo[:, 1],
            s_locate[:, 0] + s_geo[:, 2],
            s_locate[:, 1] + s_geo[:, 3]
        ], axis=-1)
        # s_detection_score = np.concatenate([s_detection, sc_score], axis=-1)
        return s_detection.reshape((-1, 2, 2))

    def draw_gt_cv_all(cls_flatten, regression_flatten, centerness_flatten, img):
        """

        :param pred_fcos_out:
        :param img:
        :return:
        """
        num_points_per_level = [len(points_per_level) for points_per_level in locations]
        for num_i, nums_level in enumerate(num_points_per_level):
            if num_i == len(num_points_per_level) - 1:
                continue
            num_points_per_level[num_i + 1] += nums_level
        cls_flatten_list = np.split(cls_flatten, num_points_per_level, axis=0)[:-1]
        regression_flatten_list = np.split(regression_flatten, num_points_per_level, axis=0)[:-1]
        centerness_flatten_list = np.split(centerness_flatten, num_points_per_level, axis=0)[:-1]

        img = img.astype(np.uint8)
        draw_img_list = []
        for f_i in range(len(cls_flatten_list)):
            draw_img_list.append(draw_gt_cv(cls_flatten_list[f_i], regression_flatten_list[f_i],
                                            centerness_flatten_list[f_i], locations[f_i]))
        draw_img_list = np.concatenate(draw_img_list, axis=0)
        for box in draw_img_list:
            draw_box = box.astype(np.int32)
            img = cv2.rectangle(img, tuple(draw_box[0]), tuple(draw_box[1]), (255, 0, 0), 2)
        img = img.astype(np.float32)
        return img

    img_draw_list = tf.py_func(draw_gt_cv_all,
                               inp=[cls_flatten, regression_flatten, centerness_flatten, in_img],
                               Tout=tf.float32)
    return img_draw_list


def tower_loss(images, input_labels, input_targets, input_mask_targets, input_mask_masks, input_mask_centers, reuse_variables=None):
    # Build inference graph
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        cls_out, target_out, mask_target_out, centerness_out = model.model(images, res_type="res50", is_training=True)

    cls_loss, boxesloss, center_loss, mask_loss = loss.total_loss(cls_out, target_out, centerness_out, mask_target_out,
                                                                  input_labels, input_targets, input_mask_targets,
                                                                  input_mask_masks, input_mask_centers)
    model_loss = boxesloss + cls_loss + center_loss + mask_loss
    total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # add summary
    if reuse_variables is None:
        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        mask_regression_flatten = []

        for tt_i in range(len(cfgs.fpn_strides)):
            _, H, W, _ = tensor_shape(cls_out[tt_i], 4)
            box_cls_flatten.append(tf.reshape(cls_out[tt_i], (-1, H * W, cfgs.num_classes)))
            box_regression_flatten.append(tf.reshape(target_out[tt_i], (-1, H * W, 4)))
            centerness_flatten.append(tf.reshape(centerness_out[tt_i], (-1, H * W, 1)))
            mask_regression_flatten.append(tf.reshape(centerness_out[tt_i], (-1, H * W, 1)))

        box_cls_flatten = tf.concat(box_cls_flatten, axis=1)
        box_regression_flatten = tf.concat(box_regression_flatten, axis=1)

        centerness_flatten = tf.concat(centerness_flatten, axis=1)
        centerness_flatten = tf.nn.sigmoid(centerness_flatten)

        new_label_level = tf.concat([tf.expand_dims(input_labels, axis=-1), input_targets, input_mask_targets], axis=-1)

        per_img_draw = draw_gt_tf(new_label_level[0], images[0])
        per_img_draw1 = draw_gt_tf(new_label_level[1], images[1])
        per_img_draw2 = draw_gt_tf(new_label_level[2], images[2])

        pred_per_img_draw = draw_pred_tf(box_cls_flatten[0], box_regression_flatten[0], centerness_flatten[0],
                                         images[0])
        pred_per_img_draw1 = draw_pred_tf(box_cls_flatten[1], box_regression_flatten[1], centerness_flatten[1],
                                          images[1])
        pred_per_img_draw2 = draw_pred_tf(box_cls_flatten[2], box_regression_flatten[2], centerness_flatten[2],
                                          images[2])
        # pred_per_img_draw1 = draw_pred_tf(fcos_pred_out[1], images[1])
        # pred_per_img_draw2 = draw_pred_tf(fcos_pred_out[2], images[2])

        # add summary
        tf.summary.image('gt_draw', tf.expand_dims(per_img_draw, axis=0))
        tf.summary.image('pred_draw', tf.expand_dims(pred_per_img_draw, axis=0))
        tf.summary.image('pred_draw1', tf.expand_dims(pred_per_img_draw1, axis=0))
        tf.summary.image('pred_draw2', tf.expand_dims(pred_per_img_draw2, axis=0))
        tf.summary.image('gt_draw1', tf.expand_dims(per_img_draw1, axis=0))
        tf.summary.image('gt_draw2', tf.expand_dims(per_img_draw2, axis=0))
        tf.summary.scalar('model_loss', model_loss)
        tf.summary.scalar('total_loss', total_loss)

    return total_loss, model_loss, cls_loss, boxesloss, center_loss, mask_loss


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    if not tf.gfile.Exists(FLAGS.checkpoint_path):
        tf.gfile.MkDir(FLAGS.checkpoint_path)
    else:
        if not FLAGS.restore:
            tf.gfile.DeleteRecursively(FLAGS.checkpoint_path)
            tf.gfile.MkDir(FLAGS.checkpoint_path)

    input_images = tf.placeholder(tf.float32, shape=[FLAGS.batch_size_per_gpu * len(gpus),
                                                     FLAGS.input_size_h, FLAGS.input_size_w, 3],
                                  name='input_images')
    input_labels = tf.placeholder(tf.float32, shape=[FLAGS.batch_size_per_gpu * len(gpus), cfgs.new_number_levels],
                                  name='input_labels')
    input_targets = tf.placeholder(tf.float32, shape=[FLAGS.batch_size_per_gpu * len(gpus), cfgs.new_number_levels, 4],
                                   name='input_targets')
    input_mask_targets = tf.placeholder(tf.float32, shape=[FLAGS.batch_size_per_gpu * len(gpus), cfgs.new_number_levels, 36],
                                        name='input_mask_targets')
    input_mask_masks = tf.placeholder(tf.float32, shape=[FLAGS.batch_size_per_gpu * len(gpus), cfgs.new_number_levels, 36],
                                      name='input_mask_masks')
    input_mask_centers = tf.placeholder(tf.float32, shape=[FLAGS.batch_size_per_gpu * len(gpus), cfgs.new_number_levels, 1],
                                        name='input_mask_centers')
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=3000, decay_rate=0.98,
                                               staircase=True)
    # learning_rate = FLAGS.learning_rate

    # warmup_steps = FLAGS.warmup_steps
    # train_steps = FLAGS.max_steps
    # learning_rate = tf.cond(
    #     pred=global_step < warmup_steps,
    #     true_fn=lambda: global_step / warmup_steps * FLAGS.learn_rate_init / 3.0,
    #     false_fn=lambda: FLAGS.learn_rate_end + 0.5 * (FLAGS.learn_rate_init - FLAGS.learn_rate_end) *
    #                      (1 + tf.cos(
    #                          (global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
    # )

    # add summary
    tf.summary.scalar('learning_rate', learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate)
    # opt = tf.train.MomentumOptimizer(learning_rate, 0.9)

    # split
    input_images_split = tf.split(input_images, len(gpus))
    input_labels_split = tf.split(input_labels, len(gpus))
    input_targets_split = tf.split(input_targets, len(gpus))
    input_mask_targets_split = tf.split(input_mask_targets, len(gpus))
    input_mask_masks_split = tf.split(input_mask_masks, len(gpus))
    input_mask_centers_split = tf.split(input_mask_centers, len(gpus))

    tower_grads = []
    reuse_variables = None
    for i, gpu_id in enumerate(gpus):
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('model_%d' % gpu_id) as scope:
                iis = input_images_split[i]
                ils = input_labels_split[i]
                its = input_targets_split[i]
                imts = input_mask_targets_split[i]
                imms = input_mask_masks_split[i]
                imcs = input_mask_centers_split[i]
                total_loss, model_loss, cls_loss, boxesloss, center_loss, mask_loss \
                    = tower_loss(iis, ils, its, imts, imms, imcs, reuse_variables)
                batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
                reuse_variables = True

                grads = opt.compute_gradients(total_loss)
                tower_grads.append(grads)

    grads = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    summary_op = tf.summary.merge_all()
    # save moving average
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # batch norm updates
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_path, tf.get_default_graph())

    init = tf.global_variables_initializer()

    if FLAGS.pretrained_model_path is not None:
        model_path = tf.train.latest_checkpoint(FLAGS.pretrained_model_path)
        variable_restore_op = slim.assign_from_checkpoint_fn(model_path,
                                                             slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        if FLAGS.restore:
            print('continue training from previous checkpoint')
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            saver.restore(sess, ckpt)
        else:
            sess.run(init)
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)

        data_generator = icdar.get_batch(num_workers=FLAGS.num_readers,
                                         input_size_w=FLAGS.input_size_w,
                                         input_size_h=FLAGS.input_size_h,
                                         batch_size=FLAGS.batch_size_per_gpu * len(gpus))

        start = time.time()
        for step in range(FLAGS.max_steps):
            data = next(data_generator)
            ml, tl, clls, bls, cls, mls, _ = sess.run([model_loss, total_loss, cls_loss, boxesloss, center_loss, mask_loss, train_op],
                                 feed_dict={input_images: data[0],
                                            input_labels: data[2],
                                            input_targets: data[3],
                                            input_mask_targets: data[4],
                                            input_mask_masks: data[5],
                                            input_mask_centers: data[6]})

            if step % 1 == 0:
                avg_time_per_step = (time.time() - start) / 1
                avg_examples_per_second = (1 * FLAGS.batch_size_per_gpu * len(gpus)) / (time.time() - start)
                start = time.time()
                print(
                    'Step {:06d}, model loss {:.4f}, total loss {:.4f}, cls loss {:.4f}, box loss {:.4f}, '
                    'center loss {:.4f}, mask loss {:.4f} {:.2f} seconds/step, {:.2f} examples/second'.format(
                        step, ml, tl, clls, bls, cls, mls, avg_time_per_step, avg_examples_per_second))

            if np.isnan(tl):
                print('Loss diverged, stop training')
                break

            if step % FLAGS.save_checkpoint_steps == 0:
                saver.save(sess, FLAGS.checkpoint_path + 'model.ckpt', global_step=global_step)

            if step % FLAGS.save_summary_steps == 0:
                _, tl, summary_str = sess.run([train_op, total_loss, summary_op],                          feed_dict={input_images: data[0],
                                            input_labels: data[2],
                                            input_targets: data[3],
                                            input_mask_targets: data[4],
                                            input_mask_masks: data[5],
                                            input_mask_centers: data[6]})
                summary_writer.add_summary(summary_str, global_step=step)


if __name__ == '__main__':
    tf.app.run()

