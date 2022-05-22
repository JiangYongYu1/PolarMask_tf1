import glob
import time
import os
import config as cfgs
import tensorflow as tf
from data_util import GeneratorEnqueuer
import numpy as np
import cv2
import json
from data import polar_utils
from pycocotools.coco import COCO
#
# tf.compat.v1.flags.DEFINE_string('training_data_path', r"/home/jyyu/data/coco/train2017/",
#                                  'training dataset to use')
# tf.compat.v1.flags.DEFINE_string('training_annotations_path',
#                                  r"/home/jyyu/data/coco/annotations/instances_train2017.json",
#                                  'training txt to use')
tf.compat.v1.flags.DEFINE_string('label_index_path',
                                 r"./coco/class_index.json",
                                 'training txt to use')  # 这个是用来做label平衡的，不同机器需重新生成，生成方法详见：coco/coco_class_index_select.py

tf.compat.v1.flags.DEFINE_string('training_data_path', r"/home/jyyu/data/cocodata/train2017/",
                                 'training dataset to use')  # 训练图片所在目录的绝对路径
tf.compat.v1.flags.DEFINE_string('training_annotations_path',
                                 r"/home/jyyu/data/cocodata/annotations/instances_train2017.json",
                                 'training txt to use')  # 训练图片的标注信息
tf.app.flags.DEFINE_integer('min_text_size', 10,
                            'if the text size is smaller than this, we ignore it during training')
tf.app.flags.DEFINE_float('min_crop_side_ratio', 0.1,
                          'when doing random crop from input image, the'
                          'min length of min(H, W')

FLAGS = tf.app.flags.FLAGS


class Coco_Seg_Dataset:
    def __init__(self, img_prefix, ann_dir):
        self.corruption = False
        self.img_prefix = img_prefix
        self.ann_dir = ann_dir
        self.img_infos = self.load_annotations(self.ann_dir)

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        # print(self.cat2label)
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info)

    def _parse_ann_info(self, ann_info, with_mask=True):
        """Parse bbox and mask annotation.
        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        self.debug = False

        if with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []

        if self.debug:
            count = 0
            total = 0
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            # filter bbox < 10
            if self.debug:
                total += 1

            if ann['area'] <= 15 or (w < 10 and h < 10) or self.coco.annToMask(ann).sum() < 15:
                # print('filter, area:{},w:{},h:{}'.format(ann['area'],w,h))
                if self.debug:
                    count += 1
                continue

            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
            if with_mask:
                gt_masks.append(self.coco.annToMask(ann))
                mask_polys = [
                    p for p in ann['segmentation'] if len(p) >= 6
                ]  # valid polygons have >= 3 points (6 coordinates)
                poly_lens = [len(p) for p in mask_polys]
                gt_mask_polys.append(mask_polys)
                gt_poly_lens.extend(poly_lens)

        if self.debug:
            print('filter:', count / total)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)

        if with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens
        return ann


def get_train_data():
    files = []
    for file_path in [FLAGS.training_data_txt_path]:
        txt_f = open(file_path, 'r')
        part_files = txt_f.readlines()
        part_files = [part_file.split("\n") for part_file in part_files]
        files += part_files
    return files


def load_annotation(train_infos: list):
    """

    :param train_infos: per train_info is str, split by ",", 0-4 box x1, y1, x2, y2,
                        5 is class id, 6 to final is mask coordinate
    :return:
    """
    boxes = []
    classes = []
    centers = []
    masks = []

    for train_info in train_infos:
        box_info = train_info.split(",")
        boxes.append(np.array(box_info[:4], dtype=np.float32).reshape((2, 2)))
        classes.append(box_info[4])

        # print(box_info[5:])
        mask_points = np.array(box_info[5:], dtype=np.float32).reshape((-1, 2))
        center = polar_utils.get_centerpoint(mask_points)

        centers.append(center)
        masks.append(mask_points)

    return np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int32), \
           np.array(centers, dtype=np.float32), masks


def load_annoataion(p):
    boxes = []
    classes = []
    if not os.path.exists(p):
        return np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int32)
    with open(p, 'r', encoding='utf-8') as f:
        reader = f.readlines()
        clses = [reade.split('\n')[0].split('\t')[1] for reade in reader]
        reader = [reade.split('\n')[0].split('\t')[0] for reade in reader]
        reader = [reade.split(',') for reade in reader]
        for line, cls in zip(reader, clses):
            x_min = float(line[0])
            y_min = float(line[1])

            x_max = float(line[2])
            y_max = float(line[3])

            boxes.append([[x_min, y_min], [x_max, y_max]])
            classes.append(cfgs.names_dict[cls])
        return np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int32)


def check_boxes(boxes, classes, xxx_todo_changeme):
    """

    :param boxes:
    :param classes:
    :param xxx_todo_changeme:
    :return:
    """
    (h, w) = xxx_todo_changeme
    if boxes.shape[0] == 0:
        return boxes, classes
    boxes[:, :, 0] = np.clip(boxes[:, :, 0], 0, w - 1)
    boxes[:, :, 1] = np.clip(boxes[:, :, 1], 0, h - 1)
    return boxes, classes


def crop_area(im, boxes, classes, crop_background=False, max_tries=50):
    '''
    make random crop from the input image
    :param im:
    :param boxes:
    :param classes:
    :return:
    '''
    h, w, _ = im.shape
    pad_h = h // 10
    pad_w = w // 10
    h_array = np.zeros((h + pad_h * 2), dtype=np.int32)
    w_array = np.zeros((w + pad_w * 2), dtype=np.int32)
    for poly in boxes:
        poly = np.round(poly, decimals=0).astype(np.int32)
        minx = np.min(poly[:, 0])
        maxx = np.max(poly[:, 0])
        w_array[minx + pad_w:maxx + pad_w] = 1
        miny = np.min(poly[:, 1])
        maxy = np.max(poly[:, 1])
        h_array[miny + pad_h:maxy + pad_h] = 1
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    if len(h_axis) == 0 or len(w_axis) == 0:
        return im, boxes, classes
    for i in range(max_tries):
        xx = np.random.choice(w_axis, size=2)
        xmin = np.min(xx) - pad_w
        xmax = np.max(xx) - pad_w
        xmin = np.clip(xmin, 0, w - 1)
        xmax = np.clip(xmax, 0, w - 1)
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy) - pad_h
        ymax = np.max(yy) - pad_h
        ymin = np.clip(ymin, 0, h - 1)
        ymax = np.clip(ymax, 0, h - 1)
        if xmax - xmin < FLAGS.min_crop_side_ratio * w or ymax - ymin < FLAGS.min_crop_side_ratio * h:
            # area too small
            continue
        # if xmax - xmin > 512 or ymax - ymin > 512:
        #     continue
        if boxes.shape[0] != 0:
            poly_axis_in_area = (boxes[:, :, 0] >= xmin) & (boxes[:, :, 0] <= xmax) \
                                & (boxes[:, :, 1] >= ymin) & (boxes[:, :, 1] <= ymax)
            selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
            selected_polys_no = np.where(np.sum(poly_axis_in_area, axis=1) == 0)[0]
            if len(selected_polys) + len(selected_polys_no) != boxes.shape[0]:
                continue
        else:
            selected_polys = []
        if len(selected_polys) == 0:
            # no text in this area
            if crop_background:
                return im[ymin:ymax + 1, xmin:xmax + 1, :], boxes[selected_polys], classes[selected_polys]
            else:
                continue
        im = im[ymin:ymax + 1, xmin:xmax + 1, :]
        boxes = boxes[selected_polys]
        classes = classes[selected_polys]

        boxes[:, :, 0] -= xmin
        boxes[:, :, 1] -= ymin
        return im, boxes, classes

    return im, boxes, classes


def resized_img(im, input_size_w, input_size_h):
    h, w, _ = im.shape

    im = cv2.resize(im, (input_size_w, input_size_h))

    ratio_h = input_size_h / h
    ratio_w = input_size_w / w

    return im, (ratio_h, ratio_w)


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


def compute_targets_for_locations(image, locations, targets, classes, masks, object_sizes_of_interest):
    """
    根据 box 和 mask的真实值，生成模型需要的形式
    :param image: 用来debug的
    :param locations:
    :param targets:
    :param classes:
    :param masks:
    :param object_sizes_of_interest:
    :return:
    """
    xs, ys = locations[:, 0], locations[:, 1]

    bboxes = targets

    labels_per_im = classes
    masks_per_im = masks

    mask_centers = []
    mask_contours = []

    # 计算重心
    for idx, mask in enumerate(masks_per_im):
        cnt, contour = polar_utils.get_single_centerpoint(mask)
        contour = contour[0]
        contour = contour.astype(np.float32)
        y, x = cnt

        mask_centers.append([x, y])
        mask_contours.append(contour)

    mask_centers = np.array(mask_centers)

    # 计算boxes的面积
    areas = (bboxes[:, 1, 0] - bboxes[:, 0, 0]) * (bboxes[:, 1, 1] - bboxes[:, 0, 1])

    # 计算网格的每个像素点到box左、上、右、下四条边的距离
    left = xs[:, np.newaxis] - bboxes[:, 0, 0]
    top = ys[:, np.newaxis] - bboxes[:, 0, 1]
    right = bboxes[:, 1, 0] - xs[:, np.newaxis]
    down = bboxes[:, 1, 1] - ys[:, np.newaxis]

    reg_targets_per_im = np.stack([left, top, right, down], axis=-1)

    # 复制数据，将他们都搞到同样的维度
    bboxes = bboxes.reshape((-1, 4))
    gt_boxes = np.repeat(np.expand_dims(bboxes, 0), len(locations), 0)

    mask_centers = np.repeat(np.expand_dims(mask_centers, 0), len(locations), 0)

    # 计算重心点附近1.5倍步长区域内符合正样本条件的点
    is_in_boxes = polar_utils.get_mask_sample_region(gt_boxes, mask_centers, cfgs.fpn_strides,
                                                     cfgs.num_points_per_level, xs, ys,
                                                     radius=cfgs.radius)
    # 都大于零，代表点在box内部
    is_in_boxes_orign = np.min(reg_targets_per_im, axis=-1) > 0

    # 根据像素点到四条边的距离得到距离的最大值，用它来判断该box适合在哪一层（P3, P4, P5, P6, P7）去预测
    max_reg_targets_per_im = np.max(reg_targets_per_im, axis=2)

    is_cared_in_the_level = (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                            (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

    locations_to_gt_area = np.repeat(np.expand_dims(areas, 0), len(locations), 0)
    locations_to_gt_area[is_in_boxes == 0] = cfgs.INF
    locations_to_gt_area[is_cared_in_the_level == 0] = cfgs.INF
    locations_to_gt_area[is_in_boxes_orign == 0] = cfgs.INF

    locations_to_gt_inds = np.argmin(locations_to_gt_area, axis=1)
    locations_to_min_aera = np.min(locations_to_gt_area, axis=1)

    reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]

    labels_per_im = labels_per_im[locations_to_gt_inds]
    labels_per_im[locations_to_min_aera == cfgs.INF] = 0

    # 计算mask的36个点的值
    mask_targets = np.zeros([len(locations), 36], dtype=np.float32)
    # 由于mask的36个角度 0 10 20 ... 350°有时候没有距离，设计这个可以排除这个的干扰，让loss更平滑
    masked_masks = np.zeros([len(locations), 36], dtype=np.float32)

    # 计算mask centers的值
    masked_centers = np.zeros([len(locations), 1], dtype=np.float32)

    # 得到选择的真值点的idx
    pos_inds = np.where(labels_per_im > 0)[0]
    pos_mask_ids = locations_to_gt_inds[pos_inds]
    # good_location = locations[pos_inds]

    # dd_img = cv2.drawKeypoints(image, cv2.KeyPoint_convert(good_location[:, np.newaxis, :]), None, color=(0, 255, 0))

    # cv2.imwrite("dd_img.jpg", dd_img)
    # cv2.imshow("dd_img", dd_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 计算每个真值点的mask_target, masked_mask, masked_center
    for p, id in zip(pos_inds, pos_mask_ids):
        x, y = locations[p]
        pos_mask_contour = np.array(mask_contours[id])
        dists, masks = polar_utils.get_36_coordinates(x, y, pos_mask_contour)
        mask_targets[p] = dists
        masked_masks[p] = masks

        # neg_mask = 1 - masks
        # neg_big_mask = neg_mask * cfgs.INF
        # neg_small_mask = -neg_mask * cfgs.INF

        l_min = np.min(dists)
        l_max = np.max(dists)

        if l_max > 0 and l_min >= 0:
            masked_centers[p] = np.sqrt(l_min / l_max)

    return labels_per_im, reg_targets_per_im, mask_targets, masked_masks, masked_centers


def get_images():
    files = []
    for file_path in [FLAGS.training_data_path]:
        for ext in ['jpg', 'png', 'jpeg', 'JPG']:
            files.extend(glob.glob(
                os.path.join(file_path, '*.{}'.format(ext))))
    return files


def resized_ann_masks(masks, input_size_w, input_size_h):
    masks = np.array(masks)
    masks = [cv2.resize(mask, (int(input_size_w), int(input_size_h))) for mask in masks]
    return np.array(masks, dtype=np.uint8)


def flip_ann_masks(masks, flip_scale):
    masks = np.array(masks)
    masks = [cv2.flip(mask, flip_scale) for mask in masks]
    return np.array(masks, dtype=np.uint8)


# def compute_mask_centerness_targets(reg_mask_targets, pos_input_mask_masks):
#     """
#
#     :param reg_mask_targets:
#     :param pos_input_mask_masks:
#     :return:
#     """
#     neg_mask = 1 - pos_input_mask_masks
#     neg_big_mask = neg_mask * cfgs.INF
#     neg_small_mask = -neg_mask * cfgs.INF
#
#     l_min = np.min(reg_mask_targets + neg_big_mask, axis=-1)
#     l_max = np.max(reg_mask_targets + neg_small_mask, axis=-1)
#
#     print(np.min(l_min), np.min(l_max))
#
#     zeros = np.zeros_like(l_min)
#     # print(np.where(np.less(l_max, 0.0)))
#
#     centerness = np.where(np.less_equal(l_max, 0.0), zeros,
#                           np.sqrt(l_min / l_max))
#     centerness = np.expand_dims(centerness, axis=-1)
#     return centerness


def generator(input_size_w=640, input_size_h=640, batch_size=8):
    """
    模型输入生成的主函数
    :param input_size_w: 输入图片的宽
    :param input_size_h: 输入图片的高
    :param batch_size: 批处理大小
    :return:
    """
    # coco数据、真值读取
    coco_data = Coco_Seg_Dataset(FLAGS.training_data_path, FLAGS.training_annotations_path)

    # 读取每个类的所对应的图片下标
    label_index_dict = json.load(open(FLAGS.label_index_path, 'r'))

    # 得到所有的类
    label_keys = list(label_index_dict.keys())

    # 得到所有的训练图片
    image_list = np.array(get_images())
    index = np.arange(0, image_list.shape[0])

    # 网格计算，用来计算网格上的每个像素点到box和mask的距离
    locations = compute_locations(cfgs.feat_shape)

    # 得到网格上每个像素点所能表示的box的大小范围
    expanded_object_sizes_of_interest = []
    for ll, points_per_level in enumerate(locations):
        object_sizes_of_interest_per_level = np.array(cfgs.object_sizes_of_interest[ll], np.float32)
        expanded_object_sizes_of_interest.append(
            np.repeat(np.expand_dims(object_sizes_of_interest_per_level, axis=0), len(points_per_level),
                      0)
        )

    expanded_object_sizes_of_interest = np.concatenate(expanded_object_sizes_of_interest, axis=0)

    points_all_level = np.concatenate(locations, axis=0)

    while True:
        # np.random.shuffle(index)
        # 初始化
        images = []
        image_fns = []

        labels = []
        reg_targets = []
        reg_mask_targets = []
        reg_mask_masks = []
        reg_centers = []

        for _ in index:
            try:
                # 随机选择一个类的一个随机图片
                i = np.random.choice(label_index_dict[np.random.choice(label_keys)])
                im_fn = os.path.join(FLAGS.training_data_path, coco_data.img_infos[i]["file_name"])

                # 读取图片，转为bgr
                im = cv2.imread(im_fn, cv2.IMREAD_UNCHANGED)
                if im is None:
                    continue
                if len(im.shape) == 2:
                    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                if im.shape[2] == 4:
                    im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)

                # 调用coco脚本，得到标注值
                ann = coco_data.get_ann_info(i)
                ann_masks = ann["masks"]
                classes = ann["labels"]
                boxes = ann["bboxes"]

                boxes = np.array(boxes).reshape((-1, 2, 2))
                if boxes.shape[0] == 0:
                    continue

                if len(boxes) != len(ann_masks):
                    ann_masks = ann_masks[:len(boxes)]

                # resize图片到指定大小
                im, (resize_ratio_3_y, resize_ratio_3_x) = resized_img(im, input_size_w, input_size_h)

                # boxes和masks按比例放缩
                boxes[:, :, 0] *= resize_ratio_3_x
                boxes[:, :, 1] *= resize_ratio_3_y

                ann_masks = resized_ann_masks(ann_masks, input_size_w, input_size_h)

                # -----------翻转---------------------------
                # 随机翻转，这里只做了左右翻转（源代码一样）
                if np.random.random() > 0.5:
                    flip_scale = 1

                    im = cv2.flip(im, flip_scale)

                    ann_masks = flip_ann_masks(ann_masks, flip_scale)

                    # if flip_scale == 1:
                    boxes[:, :, 0] = input_size_w - boxes[:, :, 0]

                    x1 = np.min(boxes[:, :, 0], axis=1)
                    x2 = np.max(boxes[:, :, 0], axis=1)

                    y1 = np.min(boxes[:, :, 1], axis=1)
                    y2 = np.max(boxes[:, :, 1], axis=1)

                    boxes = np.stack([x1, y1, x2, y2], axis=-1)
                    boxes = boxes.reshape((-1, 2, 2))
                    # if flip_scale == 0:
                    #     boxes[:, :, 1] = input_size_h - boxes[:, :, 1]
                    # if flip_scale == -1:
                    #     boxes[:, :, 0] = input_size_w - boxes[:, :, 0]
                    #     boxes[:, :, 1] = input_size_h - boxes[:, :, 1]

                # if np.random.random() > 0.5:
                #     cv2.flip()
                # for idx in range(len(masks_padded)):
                #     print(np.max(masks_padded[idx]))
                #     cv2.imshow("mask pad {}".format(idx), masks_padded[idx]*255)
                #     cv2.waitKey(0)
                #     cv2.destroyAllWindows()

                # ann_masks = masks_padded
                # 生成模型输入真实值
                labels_per_im, reg_targets_per_im, mask_targets, masked_masks, mask_centerness = \
                    compute_targets_for_locations(im, points_all_level, boxes,
                                                  classes,
                                                  ann_masks,
                                                  expanded_object_sizes_of_interest)
                # mask_centerness = compute_mask_centerness_targets(mask_targets, masked_masks)

                positive_index = np.where(labels_per_im > 0)[0]

                # print(positive_index)
                #
                if len(positive_index) <= 0:
                    print("no positive error {}".format(i))
                    continue

                if np.any(mask_centerness[positive_index]) == 0:
                    print("no ceneterness error {}".format(i))
                    continue

                # mask_get = mask_targets[positive_index]

                # for mask in mask_get:
                # #     # if np.min(mask) <= 0:
                #     print(mask)
                # # #
                # print(reg_targets_per_im[positive_index])
                # if np.min(reg_targets_per_im[positive_index]) <= 0:
                #     print("box min error {}".format(im_fn))
                #     continue

                # if np.max(reg_targets_per_im[positive_index]) >= 5000:
                #     print("box max error {}".format(im_fn))
                #     continue

                # if np.min(mask_targets[positive_index]) <= 0:
                #     print("mask min error {}".format(im_fn))
                #     continue

                if np.max(masked_masks[positive_index]) <= 0:
                    print("mask max error {}".format(im_fn))
                    continue

                labels.append(labels_per_im)
                reg_targets.append(reg_targets_per_im)
                reg_mask_targets.append(mask_targets)
                reg_mask_masks.append(masked_masks)
                reg_centers.append(mask_centerness)

                images.append(im[:, :, ::-1].astype(np.float32))
                image_fns.append(im_fn)

                if len(images) == batch_size:
                    yield images, image_fns, labels, reg_targets, reg_mask_targets, reg_mask_masks, reg_centers
                    images = []
                    image_fns = []
                    labels = []
                    reg_targets = []
                    reg_mask_targets = []
                    reg_mask_masks = []
                    reg_centers = []
            except:
                import traceback
                traceback.print_exc()
                # print(im_fn)


def get_batch(num_workers, **kwargs):
    """
    此函数作用：多进程，加快真值生成，加速模型训练
    :param num_workers:
    :param kwargs:
    :return:
    """
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        print('Generator use 10 batches for buffering, this may take a while, you can tune this yourself.')
        enqueuer.start(max_queue_size=10, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()


if __name__ == '__main__':
    data_generator = get_batch(num_workers=1)
    for step in range(50000000):
        data = next(data_generator)
        print(1)