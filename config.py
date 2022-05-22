INF = 100000000

# num_classes = 81 - 1  # coco数据集
num_classes = 81 - 1  # coco数据集

names_dict = {'person': 1, 'bicycle': 2, 'car': 3, 'motorbike': 4, 'aeroplane': 5, 'bus': 6, 'train': 7,
              'truck': 8, 'boat': 9, 'traffic light': 10, 'fire hydrant': 11, 'stop sign': 12, 'parking meter': 13,
              'bench': 14, 'bird': 15, 'cat': 16, 'dog': 17, 'horse': 18, 'sheep': 19, 'cow': 20, 'elephant': 21,
              'bear': 22, 'zebra': 23, 'giraffe': 24, 'backpack': 25, 'umbrella': 26, 'handbag': 27, 'tie': 28,
              'suitcase': 29, 'frisbee': 30, 'skis': 31, 'snowboard': 32, 'sports ball': 33, 'kite': 34,
              'baseball bat': 35, 'baseball glove': 36, 'skateboard': 37, 'surfboard': 38, 'tennis racket': 39,
              'bottle': 40, 'wine glass': 41, 'cup': 42, 'fork': 43, 'knife': 44, 'spoon': 45, 'bowl': 46,
              'banana': 47, 'apple': 48, 'sandwich': 49, 'orange': 50, 'broccoli': 51, 'carrot': 52, 'hot dog': 53,
              'pizza': 54, 'donut': 55, 'cake': 56, 'chair': 57, 'sofa': 58, 'pottedplant': 59, 'bed': 60,
              'diningtable': 61, 'toilet': 62, 'tvmonitor': 63, 'laptop': 64, 'mouse': 65, 'remote': 66,
              'keyboard': 67, 'cell phone': 68, 'microwave': 69, 'oven': 70, 'toaster': 71, 'sink': 72,
              'refrigerator': 73, 'book': 74, 'clock': 75, 'vase': 76, 'scissors': 77,
              'teddy bear': 78, 'hair drier': 79, 'toothbrush': 80}


idx_names_dict = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorbike', 5: 'aeroplane', 6: 'bus', 7: 'train',
                  8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 12: 'stop sign', 13:
                      'parking meter', 14: 'bench', 15: 'bird', 16: 'cat', 17: 'dog', 18: 'horse', 19:
                      'sheep', 20: 'cow', 21: 'elephant', 22: 'bear', 23: 'zebra', 24: 'giraffe', 25:
                      'backpack', 26: 'umbrella', 27: 'handbag', 28: 'tie', 29: 'suitcase', 30: 'frisbee',
                  31: 'skis', 32: 'snowboard', 33: 'sports ball', 34: 'kite', 35: 'baseball bat',
                  36: 'baseball glove', 37: 'skateboard', 38: 'surfboard', 39: 'tennis racket',
                  40: 'bottle', 41: 'wine glass', 42: 'cup', 43: 'fork', 44: 'knife', 45: 'spoon',
                  46: 'bowl', 47: 'banana', 48: 'apple', 49: 'sandwich', 50: 'orange', 51: 'broccoli',
                  52: 'carrot', 53: 'hot dog', 54: 'pizza', 55: 'donut', 56: 'cake', 57: 'chair',
                  58: 'sofa', 59: 'pottedplant', 60: 'bed', 61: 'diningtable', 62: 'toilet',
                  63: 'tvmonitor', 64: 'laptop', 65: 'mouse', 66: 'remote', 67: 'keyboard',
                  68: 'cell phone', 69: 'microwave', 70: 'oven', 71: 'toaster', 72: 'sink',
                  73: 'refrigerator', 74: 'book', 75: 'clock', 76: 'vase', 77: 'scissors',
                  78: 'teddy bear', 79: 'hair drier', 80: 'toothbrush'}


fpn_out_channels = 256

fpn_strides = [8, 16, 32, 64, 128]
epsilon = 10-6
img_size_h = 768
img_size_w = 1280
img_size = 1280

feat_shape = [(img_size_h // 8, img_size_w // 8), (img_size_h // 16, img_size_w // 16),
              (img_size_h // 32, img_size_w // 32), (img_size_h // 64, img_size_w // 64),
              (img_size_h // 128, img_size_w // 128)]

image_shape = (img_size_w, img_size_h)

object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF]
        ]

batch_size = 7

label_levels = [img_size_h // 8 * img_size_w // 8 * batch_size,
                img_size_h // 16 * img_size_w // 16 * batch_size,
                img_size_h // 32 * img_size_w // 32 * batch_size,
                img_size_h // 64 * img_size_w // 64 * batch_size,
                img_size_h // 128 * img_size_w // 128 * batch_size]


num_points_per_level = [img_size_h // 8 * img_size_w // 8,
                        img_size_h // 16 * img_size_w // 16,
                        img_size_h // 32 * img_size_w // 32,
                        img_size_h // 64 * img_size_w // 64,
                        img_size_h // 128 * img_size_w // 128]

radius = 1.5

new_number_levels = img_size_h // 8 * img_size_w // 8 + img_size_h // 16 * img_size_w // 16 + \
                    img_size_h // 32 * img_size_w // 32 + img_size_h // 64 * img_size_w // 64 + \
                    img_size_h // 128 * img_size_w // 128

pre_nms_thresh = 0.2
pre_nms_top_n = 1000
max_nms_n = 100

nms_th = 0.4
