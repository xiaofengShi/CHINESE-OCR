# coding:utf-8
import sys

sys.path.insert(1, "./crnn")
import torch
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import util
import dataset
import models.crnn as crnn
import keys_crnn
from math import *
import cv2

GPU = False


def dumpRotateImage_(img, degree, pt1, pt2, pt3, pt4):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    imgOut = imgRotation[int(pt1[1]):int(pt3[1]), int(pt1[0]):int(pt3[0])]
    height, width = imgOut.shape[:2]
    return imgOut


def crnnSource():
    alphabet = keys_crnn.alphabet
    converter = util.strLabelConverter(alphabet)
    if torch.cuda.is_available() and GPU:
        model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1).cuda()
    else:
        model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1).cpu()
    path = './crnn/samples/model_acc97.pth'
    model.eval()
    model.load_state_dict(torch.load(path))
    return model, converter


##加载模型
model, converter = crnnSource()


def crnnOcr(image):
    """
    crnn模型，ocr识别
    @@model,
    @@converter,
    @@im
    @@text_recs:text box

    """
    scale = image.size[1] * 1.0 / 32
    w = image.size[0] / scale
    w = int(w)
    # print "im size:{},{}".format(image.size,w)
    transformer = dataset.resizeNormalize((w, 32))
    if torch.cuda.is_available() and GPU:
        image = transformer(image).cuda()
    else:
        image = transformer(image).cpu()

    image = image.view(1, *image.size())
    image = Variable(image)
    model.eval()
    preds = model(image)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    if len(sim_pred) > 0:
        if sim_pred[0] == u'-':
            sim_pred = sim_pred[1:]

    return sim_pred
