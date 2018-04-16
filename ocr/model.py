# -*- coding: utf-8 -*-
## 修复K.ctc_decode bug 当大量测试时将GPU显存消耗完，导致错误，用decode 替代
###
import os,sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)
# from PIL import Image
import keras.backend as K

import keys_ocr
import numpy as np
from keras.layers import Flatten, BatchNormalization, Permute, TimeDistributed, Dense, Bidirectional, GRU
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Lambda
from keras.models import Model
from keras.optimizers import SGD


# from keras.models import load_model


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def get_model(height, nclass):
    rnnunit = 256
    input = Input(shape=(height, None, 1), name='the_input')
    m = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv1')(input)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(m)
    m = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='conv2')(m)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(m)
    m = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv3')(m)
    m = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv4')(m)

    m = ZeroPadding2D(padding=(0, 1))(m)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='pool3')(m)

    m = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv5')(m)
    m = BatchNormalization(axis=1)(m)
    m = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv6')(m)
    m = BatchNormalization(axis=1)(m)
    m = ZeroPadding2D(padding=(0, 1))(m)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='pool4')(m)
    m = Conv2D(512, kernel_size=(2, 2), activation='relu', padding='valid', name='conv7')(m)
    # m的输出维度为HWC?
    # 将输入的维度按照给定模式进行重排，例如，当需要将RNN和CNN网络连接时，可能会用到该层
    # 将维度转成WHC
    m = Permute((2, 1, 3), name='permute')(m)
    m = TimeDistributed(Flatten(), name='timedistrib')(m)

    m = Bidirectional(GRU(rnnunit, return_sequences=True), name='blstm1')(m)
    m = Dense(rnnunit, name='blstm1_out', activation='linear')(m)
    m = Bidirectional(GRU(rnnunit, return_sequences=True), name='blstm2')(m)
    y_pred = Dense(nclass, name='blstm2_out', activation='softmax')(m)

    basemodel = Model(inputs=input, outputs=y_pred)

    labels = Input(name='the_labels', shape=[None, ], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    model = Model(inputs=[input, labels, input_length, label_length], outputs=[loss_out])
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    # model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    # model.summary()
    return model, basemodel


characters = keys_ocr.alphabet[:]
modelPath = os.path.join(os.getcwd(), "ocr/ocr0.2.h5")
# modelPath = '/Users/xiaofeng/Code/Github/dataset/CHINESE_OCR/save_model/my_model_keras.h5'
height = 32
nclass=len(characters)+1
if os.path.exists(modelPath):
    model, basemodel = get_model(height, nclass)
    basemodel.load_weights(modelPath)
    # model.load_weights(modelPath)


def predict(im):
    """
    输入图片，输出keras模型的识别结果
    """
    im = im.convert('L')
    scale = im.size[1] * 1.0 / 32
    w = im.size[0] / scale
    w = int(w)
    im = im.resize((w, 32))
    img = np.array(im).astype(np.float32) / 255.0
    X = img.reshape((32, w, 1))
    X = np.array([X])
    y_pred = basemodel.predict(X)
    y_pred = y_pred[:, 2:, :]
    out = decode(y_pred)  ##
    # out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0])*y_pred.shape[1], )[0][0])[:, :]

    # out = u''.join([characters[x] for x in out[0]])

    if len(out) > 0:
        while out[0] == u'。':
            if len(out) > 1:
                out = out[1:]
            else:
                break

    return out


def decode(pred):
    charactersS = characters + u' '
    t = pred.argmax(axis=2)[0]
    length = len(t)
    char_list = []
    n = len(characters)
    for i in range(length):
        if t[i] != n and (not (i > 0 and t[i - 1] == t[i])):
            char_list.append(charactersS[t[i]])
    return u''.join(char_list)
