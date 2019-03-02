# -*- coding: utf-8 -*-
import dataset
import keys_keras
import numpy as np
import torch
import time
import os
import sys
sys.path.insert(0, os.getcwd())
import tensorflow as tf
import pydot
import graphviz
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import TensorBoard
from keras.utils import plot_model

characters = keys_keras.alphabet[:]
from model import get_model
nclass = len(characters) + 1
trainroot = '../data/lmdb/train'
valroot = '../data/lmdb/val'
# modelPath = '../pretrain-models/keras.hdf5'
modelPath = '/Users/xiaofeng/Code/Github/dataset/CHINESE_OCR/save_model/my_model_keras.h5'
workers = 4
imgH = 32
imgW = 256
keep_ratio = False
random_sample = False
batchSize = 32
testSize = 16
n_len = 50
loss = 1000
interval = 50
LEARNING_RATE = 0.01
Learning_decay_step = 20000
PERCEPTION = 0.3
EPOCH_NUMS = 1000000
MODEL_PATH = '/Users/xiaofeng/Code/Github/dataset/CHINESE_OCR/save_model'
LOG_FILE = 'log.txt'
SUMMARY_PATH = './log/'
if not os.path.exists(MODEL_PATH):
    print('Creating save model path!!')
    os.makedirs(MODEL_PATH)
if not os.path.exists(SUMMARY_PATH):
    os.makedirs(SUMMARY_PATH)

model, basemodel = get_model(
    height=imgH, nclass=nclass, learning_rate=LEARNING_RATE)

config = tf.ConfigProto(intra_op_parallelism_threads=2)
config.gpu_options.per_process_gpu_memory_fraction = PERCEPTION
KTF.set_session(tf.Session(config=config))

# 加载预训练参数
if os.path.exists(modelPath):
    # basemodel.load_weights(modelPath)
    model.load_weights(modelPath)

plot_model(basemodel, to_file='basemodel.png')
plot_model(model, to_file='model.png')


def one_hot(text, length=10, characters=characters):
    label = np.zeros(length)
    for i, char in enumerate(text):
        index = characters.find(char)
        if index == -1:
            index = characters.find(u' ')
        if i < length:
            label[i] = index
    return label


# 导入数据
if random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, batchSize)
else:
    sampler = None
train_dataset = dataset.lmdbDataset(root=trainroot, target_transform=one_hot)
# print(len(train_dataset))

test_dataset = dataset.lmdbDataset(
    root=valroot,
    transform=dataset.resizeNormalize((imgW, imgH)),
    target_transform=one_hot)

# 生成训练用数据
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batchSize,
    shuffle=True,
    sampler=sampler,
    num_workers=int(workers),
    collate_fn=dataset.alignCollate(
        imgH=imgH, imgW=imgW, keep_ratio=keep_ratio))

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=testSize, shuffle=True, num_workers=int(workers))

j = 0
print('Strat training!!')
for i in range(EPOCH_NUMS):
    for X, Y in train_loader:
        start = time.time()
        X = X.numpy()
        X = X.reshape((-1, imgH, imgW, 1))
        Y = np.array(Y)
        Length = int(imgW / 4) - 2
        batch = X.shape[0]
        X_train, Y_train = [X, Y,
                            np.ones(batch) * Length,
                            np.ones(batch) * n_len], np.ones(batch)
        print('IMG_SHAPE:', np.shape(X))
        print('LABEL_SHAPE:', np.shape(Y))
        # print(np.shape(X_train))
        model.train_on_batch(X_train, Y_train)
        if j % interval == 0:
            times = time.time() - start
            currentLoss_train = model.evaluate(X_train, Y_train)
            X, Y = next(iter(test_loader))
            X = X.numpy()
            X = X.reshape((-1, imgH, imgW, 1))
            Y = Y.numpy()
            Y = np.array(Y)
            batch = X.shape[0]
            X_val, Y_val = [
                X, Y, np.ones(batch) * Length,
                np.ones(batch) * n_len], np.ones(batch)
            crrentLoss = model.evaluate(X_val, Y_val)
            print('Learning rate is: ', LEARNING_RATE)
            now_time = time.strftime('%Y/%m/%d-%H:%M:%S',
                                     time.localtime(time.time()))
            print('Time: [%s]--Step/Epoch/Total: [%d/%d/%d]' % (now_time, j, i,
                                                                EPOCH_NUMS))
            print('\tTraining Loss is: [{}]'.format(currentLoss_train))
            print('\tVal Loss is: [{}]'.format(crrentLoss))
            print('\tSpeed is: [{}] Samples/Secs'.format(interval / times))
            path = MODEL_PATH + '/my_model_keras.h5'
            with open(LOG_FILE, mode='a') as log_file:
                log_str = now_time + '----global_step:' + str(
                    j) + '----loss:' + str(loss) + '\n'
                log_file.writelines(log_str)
            log_file.close()
            print('\tWriting to the file: log.txt')
            print("\tSave model to disk: {}".format(path))
            model.save(path)
            if crrentLoss < loss:
                loss = crrentLoss
        if j > 0 and j % Learning_decay_step == 0:
            LEARNING_RATE_ori = LEARNING_RATE
            LEARNING_RATE = 0.5 * LEARNING_RATE
            print('\tUpdating Leaning rate from {} to {}'.format(
                LEARNING_RATE_ori, LEARNING_RATE))
        j += 1
