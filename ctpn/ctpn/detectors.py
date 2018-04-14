# coding:utf-8
import sys

import numpy as np

from .cfg import Config as cfg
from .other import normalize

sys.path.append('..')
from ..lib.fast_rcnn.nms_wrapper import nms
# from lib.fast_rcnn.test import  test_ctpn

from .text_proposal_connector import TextProposalConnector


class TextDetector:
    """
        Detect text from an image
    """

    def __init__(self):
        """
        pass
        """
        self.text_proposal_connector = TextProposalConnector()

    def detect(self, text_proposals, scores, size):
        """
        Detecting texts from an image
        :return: the bounding boxes of the detected texts
        """
        # text_proposals, scores=self.text_proposal_detector.detect(im, cfg.MEAN)
        keep_inds = np.where(scores > cfg.TEXT_PROPOSALS_MIN_SCORE)[0]
        text_proposals, scores = text_proposals[keep_inds], scores[keep_inds]

        sorted_indices = np.argsort(scores.ravel())[::-1]
        text_proposals, scores = text_proposals[sorted_indices], scores[sorted_indices]

        # nms for text proposals
        keep_inds = nms(np.hstack((text_proposals, scores)), cfg.TEXT_PROPOSALS_NMS_THRESH)
        text_proposals, scores = text_proposals[keep_inds], scores[keep_inds]

        scores = normalize(scores)

        text_lines = self.text_proposal_connector.get_text_lines(text_proposals, scores, size)

        keep_inds = self.filter_boxes(text_lines)
        text_lines = text_lines[keep_inds]

        if text_lines.shape[0] != 0:
            keep_inds = nms(text_lines, cfg.TEXT_LINE_NMS_THRESH)
            text_lines = text_lines[keep_inds]

        return text_lines

    def filter_boxes(self, boxes):
        heights = boxes[:, 3] - boxes[:, 1] + 1
        widths = boxes[:, 2] - boxes[:, 0] + 1
        scores = boxes[:, -1]
        return np.where((widths / heights > cfg.MIN_RATIO) & (scores > cfg.LINE_MIN_SCORE) &
                        (widths > (cfg.TEXT_PROPOSALS_WIDTH * cfg.MIN_NUM_PROPOSALS)))[0]
