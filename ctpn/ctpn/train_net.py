#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _Author_: xiaofeng
# Date: 2018-04-16 10:55:15
# Last Modified by: xiaofeng
# Last Modified time: 2018-04-16 10:55:15
'''
使用keras进行网络训练，速度相对pytorch比较慢
'''
import os.path as osp
import pprint
import sys
import os

# sys.path.append(os.getcwd())
# this_dir = os.path.dirname(__file__)
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)

from lib.fast_rcnn.train import get_training_roidb, train_net
from lib.fast_rcnn.config import cfg_from_file, get_output_dir, get_log_dir
from lib.datasets.factory import get_imdb
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg

if __name__ == '__main__':
    # 将text.yml的配置与默认config中的默认配置进行合并
    cfg_from_file('text.yml')
    print('Using config:~~~~~~~~~~~~~~~~')
    # 根据给定的名字，得到要加载的数据集
    imdb = get_imdb('voc_2007_trainval')
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    # 准备训练数据
    roidb = get_training_roidb(imdb)
    # 模型输出的路径
    output_dir = get_output_dir(imdb, None)
    # summary的输出路径
    log_dir = get_log_dir(imdb)
    print('Output will be saved to `{:s}`'.format(output_dir))
    print('Logs will be saved to `{:s}`'.format(log_dir))

    device_name = '/gpu:0'
    print(device_name)

    network = get_network('VGGnet_train')

    train_net(
        network,
        imdb,
        roidb,
        output_dir=output_dir,
        log_dir=log_dir,
        # pretrained_model=
        # '/Users/xiaofeng/Code/Github/dataset/CHINESE_OCR/ctpn/pretrain/VGG_imagenet.npy',
        # pretrained_model='/home/xiaofeng/data/ctpn/pretrainde_vgg',
        pretrained_model=None,
        max_iters=180000,
        restore=bool(int(1)))
