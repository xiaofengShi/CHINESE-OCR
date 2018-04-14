import sys, os

import tensorflow as tf

from .cfg import Config
from .other import resize_im

sys.path.append(os.getcwd())
from lib.fast_rcnn.config import cfg
from lib.networks.factory import get_network
from lib.fast_rcnn.test import test_ctpn

# from ..lib.networks.factory import get_network
# from ..lib.fast_rcnn.config import cfg
# from..lib.fast_rcnn.test import test_ctpn


def load_tf_model():
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    # load network
    net = get_network("VGGnet_test")
    # load model
    saver = tf.train.Saver()
    # sess = tf.Session(config=config)
    sess = tf.Session()
    ckpt = tf.train.get_checkpoint_state('/Users/xiaofeng/Code/Github/dataset/CHINESE_OCR/ctpn/checkpoints/')
    reader = tf.train.NewCheckpointReader(ckpt.model_checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print("tensor_name: ", key)
        # print(reader.get_tensor(key))
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("load vggnet done")
    return sess, saver, net


##init model
sess, saver, net = load_tf_model()


def ctpn(img):
    """
    text box detect
    """
    scale, max_scale = Config.SCALE, Config.MAX_SCALE
    img, f = resize_im(img, scale=scale, max_scale=max_scale)
    scores, boxes = test_ctpn(sess, net, img)
    return scores, boxes, img
