import cv2
import os
import config as cfgs
import numpy as np
import tensorflow as tf
import math
from tqdm import tqdm


tf.app.flags.DEFINE_string('test_data_path', '//home/jyyu/data/cocodata/val2017/', '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', '/home/jyyu/models/cocoapi-master/model_files/polarmask/res50_final/polar_mask_res50.pb', '')
tf.app.flags.DEFINE_string('output_path', '/home/jyyu/data/cocodata/pre_img_fcos/', '')

FLAGS = tf.app.flags.FLAGS


def get_images():
    """
    find the image files in the test data path
    :return:
    """
    files = []
    exts = ['jpeg', 'jpg', 'png', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
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


def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list

    polar_graph = tf.Graph()
    pb_f = tf.gfile.FastGFile(FLAGS.checkpoint_path, 'rb')
    serialized_graph = pb_f.read()
    frozen_graph_def = tf.GraphDef()
    frozen_graph_def.ParseFromString(serialized_graph)
    with polar_graph.as_default():
        tf.import_graph_def(frozen_graph_def, name='')

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    sess = tf.Session(graph=polar_graph, config=tf.ConfigProto(gpu_options=gpu_options))
    pred_boxes, pred_cls_scores, pred_cls_labels, pred_masks = sess.graph.get_tensor_by_name('boxes_concat:0'), \
                                  sess.graph.get_tensor_by_name('cls_scores_concat:0'), \
                                  sess.graph.get_tensor_by_name('cls_labels_concat:0'), \
                                  sess.graph.get_tensor_by_name('select_masks_concat:0')
    input_images = sess.graph.get_tensor_by_name('input_images:0')

    im_fn_list = get_images()

    idx = 0
    for im_fn in tqdm(im_fn_list):
        im = cv2.imread(im_fn, cv2.IMREAD_UNCHANGED)
        if len(im.shape) == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        if im.shape[-1] == 4:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)

        im = im[:, :, ::-1]
        im_resized, (ratio_h, ratio_w) = resized_img(im, cfgs.img_size_w, cfgs.img_size_h)

        # h, w = im_resized.shape[:2]
        # im_padded = np.zeros((cfgs.img_size, cfgs.img_size, 3), dtype=np.uint8)
        # im_padded[:h, :w, :] = im_resized.copy()
        # im_resized = im_padded

        out_cls_labels, out_cls_scores, out_boxes, out_masks = sess.run([pred_cls_labels,
                                                                         pred_cls_scores,
                                                                         pred_boxes,
                                                                         pred_masks],
                                                                        feed_dict={input_images: [im_resized]})

        out_boxes = out_boxes.reshape((-1, 2, 2))

        out_boxes[:, :, 0] /= ratio_h
        out_boxes[:, :, 1] /= ratio_w

        out_masks[:, :, 0] /= ratio_w
        out_masks[:, :, 1] /= ratio_h

        im_resized = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        for label, score, box, mask_point in zip(out_cls_labels, out_cls_scores, out_boxes, out_masks):
            draw_box = box.astype(np.int).reshape((2, 2))
            draw_mask_point = mask_point.astype(np.int)

            name_c = cfgs.idx_names_dict[label + 1]
            # print(name_c, score)
            im_resized = cv2.putText(im_resized, name_c,
                                     (draw_box[0][1], draw_box[0][0]), cv2.FONT_HERSHEY_SIMPLEX,
                                     0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            im_resized = cv2.rectangle(im_resized, (draw_box[0][1], draw_box[0][0]),
                                       (draw_box[1][1], draw_box[1][0]),
                                       (255, 0, 0), 2)

            im_resized = cv2.drawKeypoints(im_resized, cv2.KeyPoint_convert(draw_mask_point[:, np.newaxis, :]),
                                           None, color=(0, 255, 0))
        cv2.imwrite("./catch/image_{}.jpg".format(idx), im_resized)
        idx += 1
                # print("out_cls_labels: {}".format(out_cls_labels.shape))
                # print("out_cls_scores: {}".format(out_cls_scores.shape))
                # print("out_boxes: {}".format(out_boxes.shape))
                # print("out_masks: {}".format(out_masks.shape))


if __name__ == '__main__':
    tf.app.run()








