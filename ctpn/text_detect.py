import numpy as np
# import tensorflow as tf
from .ctpn.detectors import TextDetector
from .ctpn.model import ctpn
from .ctpn.other import draw_boxes


def text_detect(img):
    scores, boxes, img = ctpn(img)
    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    text_recs, tmp = draw_boxes(img, boxes, caption='im_name', wait=True, is_display=False)
    return text_recs, tmp, img
