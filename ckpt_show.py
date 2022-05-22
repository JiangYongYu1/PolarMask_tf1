# -*- coding: utf-8 -*-
# @Time    : 2020/4/14 16:07
# @Author  : JiangYong Yu
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# ckpt_path = '/data/yujiangyong/Center-JianFan_model_files/db_tf/save_best/model_final.ckpt'

g = tf.Graph()
with g.as_default() as g:
    tf.train.import_meta_graph(r'/home/jyyu/work/model_files/polar_mask/res18_final/polar_mask_res18.ckpt.meta')

with tf.Session(graph=g) as sess:
    filw_writer = tf.summary.FileWriter(logdir=r'/home/jyyu/work/model_files/polar_mask/res18_final/', graph=g)