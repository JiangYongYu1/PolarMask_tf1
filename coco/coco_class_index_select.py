import numpy as np
from pycocotools.coco import COCO
import json
# import config
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


class CocoSegDataset:
    def __init__(self, img_prefix, ann_dir):
        self.corruption = False
        self.img_prefix = img_prefix
        self.ann_dir = ann_dir
        self.img_infos = self.load_annotations(self.ann_dir)
        self.label_index_dict = dict()

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
        return self._parse_ann_info(ann_info, idx)

    def _parse_ann_info(self, ann_info, index):
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue

            label = self.cat2label[ann['category_id']]

            if label not in self.label_index_dict:
                self.label_index_dict[label] = []
            self.label_index_dict[label].append(index)


train_dataset = CocoSegDataset(img_prefix="/home/jyyu/data/cocodata/train2017/",
                               ann_dir="/home/jyyu/data/cocodata/annotations/instances_train2017.json")

image_length = len(train_dataset.img_infos)

for i in range(image_length):
    train_dataset.get_ann_info(i)


train_class_index = dict()
train_class_index_show = dict()

for key, value in train_dataset.label_index_dict.items():
    # print("key {}, value {}".format(idx_names_dict[key], len(set(value))))
    train_class_index[key] = list(set(value))
    train_class_index_show[idx_names_dict[key]] = len(set(value))

json.dump(train_class_index, open("class_index.json", "w"))
json.dump(train_class_index_show, open("class_index_show.json", "w"))

