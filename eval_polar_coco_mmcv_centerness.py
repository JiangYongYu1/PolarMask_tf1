import cv2
import time
import os
import config as cfgs
import tensorflow as tf
import math
from tqdm import tqdm
import numpy as np
import pycocotools.mask as mask_util
from coco.coco_utils import coco_eval, results2json
from pycocotools.coco import COCO


tf.app.flags.DEFINE_string('val_data_path', '/home/jyyu/data/cocodata/val2017/', '')
tf.app.flags.DEFINE_string('val_ann_path', '/home/jyyu/data/cocodata/annotations/instances_val2017.json', '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', '/home/jyyu/models/cocoapi-master/model_files/polarmask/checkpoint_path_big/', '')
tf.app.flags.DEFINE_string('output_path', '/home/jyyu/data/cocodata/pre_img_fcos/', '')

import model

FLAGS = tf.app.flags.FLAGS


def get_images():
    """
    find the image files in the test data path
    :return:
    """
    files = []
    exts = ['jpeg', 'jpg', 'png', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.val_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def resized_img(im, input_size_w, input_size_h):
    h, w, _ = im.shape

    im = cv2.resize(im, (input_size_w, input_size_h))

    ratio_h = input_size_h / h
    ratio_w = input_size_w / w

    return im, (ratio_h, ratio_w)


def compute_locations_per_level_tf(h, w, stride):
    x_centers = (tf.range(w, dtype=tf.float32) + 0.5) * stride
    y_centers = (tf.range(h, dtype=tf.float32) + 0.5) * stride
    x_centers, y_centers = tf.meshgrid(x_centers, y_centers)

    y_centers = tf.reshape(y_centers, (-1, ))
    x_centers = tf.reshape(x_centers, (-1, ))

    locations = tf.stack((x_centers, y_centers), axis=-1)
    return locations


def compute_locations_tf(features):
    locations = []
    for level, feature in enumerate(features):
        h, w = feature
        locations_per_level = compute_locations_per_level_tf(
            h, w, cfgs.fpn_strides[level])
        locations.append(locations_per_level)
    return locations


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


def forward_for_single_feature_map(locations, box_cls, centerness, box_regression, mask_regression):
    """

    :param locations:
    :param box_cls:
    :param centerness:
    :param box_regression:
    :param mask_regression:
    :return:
    """
    N, H, W, C = box_cls.shape

    box_cls = tf.reshape(box_cls, (N, -1, cfgs.num_classes))

    centerness = tf.reshape(centerness, (N, -1, 1))

    box_regression = tf.reshape(box_regression, (N, -1, 4))
    mask_regression = tf.reshape(mask_regression, (N, -1, 36))

    final_cls = tf.reduce_max(box_cls * centerness, axis=-1)

    results = []
    for i in range(N):
        per_final_cls = final_cls[i]

        # idx_select = tf.where(per_final_cls > 0.5)

        # per_final_cls = tf.gather_nd(per_final_cls, idx_select)

        per_box_cls = box_cls[i]
        # per_box_cls = tf.gather_nd(per_box_cls, idx_select)

        per_centerness = centerness[i]
        # per_centerness = tf.gather_nd(per_centerness, idx_select)

        per_mask_regression = mask_regression[i]
        # per_mask_regression = tf.gather_nd(per_mask_regression, idx_select)

        per_box_regression = box_regression[i]
        # per_box_regression = tf.gather_nd(per_box_regression, idx_select)

        # per_locations = tf.gather_nd(locations, idx_select)
        # print(per_box_regression.shape)
        select_numbers, _ = tensor_shape(per_box_regression, 2)
        nms_pre = tf.reduce_min([select_numbers, cfgs.pre_nms_top_n])
        _, top_idx = tf.nn.top_k(per_final_cls, nms_pre, True)

        top_idx = tf.expand_dims(top_idx, axis=1)

        # per_box_cls = tf.gather_nd(per_box_cls, top_idx)
        per_box_cls = tf.gather_nd(per_box_cls, top_idx)
        per_centerness = tf.gather_nd(per_centerness, top_idx)
        per_mask_regression = tf.gather_nd(per_mask_regression, top_idx)
        per_box_regression = tf.gather_nd(per_box_regression, top_idx)
        per_locations = tf.gather_nd(locations, top_idx)

        detections = tf.stack([
            per_locations[:, 0] - per_box_regression[:, 0],
            per_locations[:, 1] - per_box_regression[:, 1],
            per_locations[:, 0] + per_box_regression[:, 2],
            per_locations[:, 1] + per_box_regression[:, 3],
        ], axis=-1)
        #

        angle_stride = tf.range(0, 360, 10, dtype=tf.float32)
        sin_cos = tf.stack([tf.cos(angle_stride * math.pi / 180), tf.sin(angle_stride * math.pi / 180)], axis=-1)
        s_mask_delta = tf.expand_dims(sin_cos, axis=0) * tf.expand_dims(per_mask_regression, axis=2)

        s_mask_points = tf.expand_dims(per_locations, axis=1) + s_mask_delta[..., ::-1]

        detections = tf.reshape(detections, (-1, 4))
        detections = tf.stack([detections[:, 1], detections[:, 0], detections[:, 3], detections[:, 2]], axis=1)

        results.append([per_box_cls, per_centerness, detections, s_mask_points])

    return results


def polar_process_tf(locations, cls_out, centerness_out, box_regression, mask_regression):
    sampled_boxes = []
    for _, (l, cs, cn, b, m) in enumerate(zip(locations, cls_out, centerness_out, box_regression, mask_regression)):
        sampled_boxes.append(
            forward_for_single_feature_map(
                l, cs, cn, b, m
            )
        )

    simple_cls = []
    simple_center = []
    simple_batch = []
    simple_mask_batch = []

    for lever_boxes in sampled_boxes:
        simple_cls.append(lever_boxes[0][0])
        simple_center.append(lever_boxes[0][1])
        simple_batch.append(lever_boxes[0][2])
        simple_mask_batch.append(lever_boxes[0][3])

    all_cls = tf.concat(simple_cls, axis=0)
    all_center = tf.concat(simple_center, axis=0)
    all_boxes = tf.concat(simple_batch, axis=0)
    all_masks = tf.concat(simple_mask_batch, axis=0)

    all_select_boxes = []
    all_select_cls_scores = []
    all_select_cls_labels = []
    all_select_masks = []

    for i in range(cfgs.num_classes):
        class_cls = all_cls[:, i] * all_center[:, 0]
        # class_cls = all_cls[:, i]
        selected_indices = tf.image.non_max_suppression(boxes=all_boxes, scores=class_cls,
                                                        max_output_size=cfgs.max_nms_n,
                                                        iou_threshold=cfgs.nms_th,
                                                        score_threshold=cfgs.pre_nms_thresh)

        selected_nums = tensor_shape(selected_indices, 1)

        if selected_nums == 0:
            continue

        selected_indices = tf.expand_dims(selected_indices, axis=1)

        select_boxes = tf.gather_nd(all_boxes, selected_indices)
        all_select_boxes.append(select_boxes)

        select_cls_scores = tf.gather_nd(class_cls, selected_indices)
        all_select_cls_scores.append(select_cls_scores)

        select_cls_labels = tf.ones_like(select_cls_scores) * i
        all_select_cls_labels.append(select_cls_labels)

        selected_masks = tf.gather_nd(all_masks, selected_indices)
        all_select_masks.append(selected_masks)

    all_select_boxes = tf.concat(all_select_boxes, axis=0)
    all_select_cls_scores = tf.concat(all_select_cls_scores, axis=0)
    all_select_cls_labels = tf.concat(all_select_cls_labels, axis=0)
    all_select_masks = tf.concat(all_select_masks, axis=0)

    return all_select_cls_labels, all_select_cls_scores, all_select_boxes, all_select_masks


def box_mask2result(boxes, masks, labels, num_classes, image_h, image_w):
    """

    :param boxes: shape (n, 5)
    :param masks: shape(n, 36, 2)
    :param labels: shape(n, )
    :param image_h: image h
    :param image_w: image w
    :param num_classes: class number, don't including background class
    :return:
    """
    mask_results = [[] for _ in range(num_classes)]

    for i in range(masks.shape[0]):
        im_mask = np.zeros((image_h, image_w), dtype=np.uint8)
        mask = [np.expand_dims(masks[i], axis=1).astype(np.int)]
        im_mask = cv2.drawContours(im_mask, mask, -1, 1, -1)
        rle = mask_util.encode(
            np.array(im_mask[:, :, np.newaxis], order='F'))[0]
        label = labels[i]
        mask_results[label].append(rle)

    if boxes.shape[0] == 0:
        bbox_results = [
            np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)
        ]
        return bbox_results, mask_results
    else:
        bbox_results = [boxes[labels == i, :] for i in range(num_classes)]
        return bbox_results, mask_results


class CocoSegDataset:
    def __init__(self, img_prefix, ann_dir):
        self.corruption = False
        self.img_prefix = img_prefix
        self.ann_dir = ann_dir
        self.load_annotations(self.ann_dir)

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        # print(self.cat2label)
        self.img_ids = self.coco.getImgIds()

    def __len__(self):
        return len(self.img_ids)


def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[1, cfgs.img_size_h, cfgs.img_size_w, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        logits, bbox_reg, mask_reg, centerness_list = model.model(input_images, res_type="res50", is_training=False)

        locations = compute_locations_tf(cfgs.feat_shape)

        pred_cls_labels, pred_cls_scores, pred_boxes, pred_masks = polar_process_tf(locations,
                                                                                    logits,
                                                                                    centerness_list,
                                                                                    bbox_reg,
                                                                                    mask_reg)
        variable_averages = tf.train.ExponentialMovingAverage(0.999, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())
        if not os.path.exists(FLAGS.output_path):
            os.mkdir(FLAGS.output_path)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            val_coco_data = CocoSegDataset(FLAGS.val_data_path, FLAGS.val_ann_path)

            print(len(val_coco_data))

            # im_fn_list = get_images()
            image_size = len(val_coco_data.img_ids)
            results = []

            for index in tqdm(range(image_size)):
                image_info = val_coco_data.coco.loadImgs(val_coco_data.img_ids[index])[0]
                im_fn = os.path.join(FLAGS.val_data_path, image_info["file_name"])
                im = cv2.imread(im_fn, cv2.IMREAD_UNCHANGED)
                if len(im.shape) == 2:
                    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                if im.shape[-1] == 4:
                    im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)

                origin_h, origin_w = im.shape[:2]

                im = im[:, :, ::-1]
                im_resized, (ratio_h, ratio_w) = resized_img(im, cfgs.img_size_w, cfgs.img_size_h)

                out_cls_labels, out_cls_scores, out_boxes, out_masks = sess.run([pred_cls_labels,
                                                                                 pred_cls_scores,
                                                                                 pred_boxes,
                                                                                 pred_masks],
                                                                                feed_dict={input_images: [im_resized]})

                out_cls_scores = np.sqrt(np.sqrt(np.sqrt(out_cls_scores)))

                out_boxes = out_boxes[:, [1, 0, 3, 2]].reshape((-1, 2, 2))

                out_boxes[:, :, 0] /= ratio_w
                out_boxes[:, :, 1] /= ratio_h

                out_masks[:, :, 0] /= ratio_w
                out_masks[:, :, 1] /= ratio_h

                out_boxes = out_boxes.reshape((-1, 4))
                out_boxes = np.concatenate([out_boxes, out_cls_scores[..., np.newaxis]], axis=-1)

                result = box_mask2result(out_boxes, out_masks, out_cls_labels.astype(np.int),
                                         cfgs.num_classes, origin_h, origin_w)

                results.append(result)

            result_files = results2json(val_coco_data, results, "val")
            eval_types = ["segm", "bbox"]
            coco_eval(result_files, eval_types, val_coco_data.coco)



                # im_resized = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                # for label, score, box, mask_point in zip(out_cls_labels, out_cls_scores, out_boxes, out_masks):
                #     draw_box = box.astype(np.int).reshape((2, 2))
                #     draw_mask_point = mask_point.astype(np.int)
                #
                #     name_c = cfgs.idx_names_dict[label + 1]
                #
                #     im_resized = cv2.putText(im_resized, name_c,
                #                              (draw_box[0][1], draw_box[0][0]), cv2.FONT_HERSHEY_SIMPLEX,
                #                              0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
                #     im_resized = cv2.rectangle(im_resized, (draw_box[0][1], draw_box[0][0]),
                #                                (draw_box[1][1], draw_box[1][0]),
                #                                (255, 0, 0), 2)
                #
                #     im_resized = cv2.drawKeypoints(im_resized, cv2.KeyPoint_convert(draw_mask_point[:, np.newaxis, :]),
                #                                    None, color=(0, 255, 0))
                # cv2.imwrite("image.jpg", im_resized)
                # print("out_cls_labels: {}".format(out_cls_labels.shape))
                # print("out_cls_scores: {}".format(out_cls_scores.shape))
                # print("out_boxes: {}".format(out_boxes.shape))
                # print("out_masks: {}".format(out_masks.shape))


if __name__ == '__main__':
    tf.app.run()








