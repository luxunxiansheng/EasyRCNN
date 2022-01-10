# #### BEGIN LICENSE BLOCK #####
# MIT License
#    
# Copyright (c) 2021 Bin.Li (ornot2008@yahoo.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# #### END LICENSE BLOCK #####
# /

"""
Default trainning configurations for Faster R-CNN.
"""
import os

from yacs.config import CfgNode as ConfigNode

# -----------------------ROOT ----------------------------------------------------- #
_C = ConfigNode()

# ----------------------- TEST ----------------------------------------------------#
_C.TEST = ConfigNode()

# ----------------------- LOG ----------------------------------------------------- #
_C.LOG = ConfigNode()
_C.LOG.LOG_DIR = '/media/yan/D/ornot/workspace/object_detection/log'

# ----------------------- CHECKPOINT-----------------------------------------------#
_C.CHECKPOINT = ConfigNode()
_C.CHECKPOINT.CHECKPOINT_PATH = '/media/yan/D/ornot/workspace/object_detection/checkpoint/model_checkpoint.ckpt'

# ----------------------- DATASET ------------------------------------------------- #
_C.VOC_DATASET = ConfigNode()
_C.VOC_DATASET.DATA_DIR ='/home/yan/data/VOCdevkit/VOC2007'
_C.VOC_DATASET.USE_DIFFICULT_LABEL = False
_C.VOC_DATASET.AUGMENTED = False
_C.VOC_DATASET.MIN_SIZE = 600
_C.VOC_DATASET.MAX_SIZE = 1000

# -------------------------- RPN ---------------------------------------------------#
_C.RPN = ConfigNode()
_C.RPN.FEATURE_CHANNELS = 512
_C.RPN.MID_CHANNELS = 512
_C.RPN.RPN_SIGMA = 3  

# ---------------------------- RPN.ANCHOR_CREATOR-----------------------------------#
_C.RPN.ANCHOR_CREATOR = ConfigNode()
_C.RPN.ANCHOR_CREATOR.ANCHOR_RATIOS = [0.5, 1, 2]
_C.RPN.ANCHOR_CREATOR.ANCHOR_SCALES = [8, 16, 32]
_C.RPN.ANCHOR_CREATOR.FEATURE_STRIDE = 16

#-----------------------------RPN.ANCHOR_TARGET_CREATOR-----------------------------#
_C.RPN.ANCHOR_TARGET_CREATOR = ConfigNode()
_C.RPN.ANCHOR_TARGET_CREATOR.N_SAMPLES = 256
_C.RPN.ANCHOR_TARGET_CREATOR.POSITIVE_RATIO = 0.5
_C.RPN.ANCHOR_TARGET_CREATOR.NEGATIVE_IOU_THRESHOLD = 0.3
_C.RPN.ANCHOR_TARGET_CREATOR.POSITIVE_IOU_THRESHOLD = 0.7

# -----------------------------RPN.PROPOAL_CREATOR--------------------#
_C.RPN.PROPOSAL_CREATOR = ConfigNode()
_C.RPN.PROPOSAL_CREATOR.NMS_THRESHOLD = 0.7
_C.RPN.PROPOSAL_CREATOR.N_PRE_NMS = 12000
_C.RPN.PROPOSAL_CREATOR.N_POST_NMS = 2000
_C.RPN.PROPOSAL_CREATOR.MIN_SIZE = 16

# -----------------------RPN.PROPOAL_TARGET_CREATOR---------------#
_C.RPN.PROPOSAL_TARGET_CREATOR = ConfigNode()
_C.RPN.PROPOSAL_TARGET_CREATOR.N_SAMPLES = 128
_C.RPN.PROPOSAL_TARGET_CREATOR.POSITIVE_RATIO = 0.25
_C.RPN.PROPOSAL_TARGET_CREATOR.POSITIVE_IOU_THRESHOLD = 0.5
_C.RPN.PROPOSAL_TARGET_CREATOR.NEGATIVE_IOU_THRESHOLD_HI = 0.5
_C.RPN.PROPOSAL_TARGET_CREATOR.NEGATIVE_IOU_THRESHOLD_LO = 0.0
_C.RPN.PROPOSAL_TARGET_CREATOR.OFFSET_NORM_MEAN = [0.0, 0.0, 0.0, 0.0]
_C.RPN.PROPOSAL_TARGET_CREATOR.OFFSET_NORM_STD = [0.1, 0.1, 0.2, 0.2]


# ----------------------- FAST_RCNN -----------------------------------------------------#
_C.FAST_RCNN = ConfigNode()
_C.FAST_RCNN.IN_CHANNELS = _C.RPN.FEATURE_CHANNELS
_C.FAST_RCNN.FC7_CHANNELS = 4096
_C.FAST_RCNN.NUM_CLASSES = 20
_C.FAST_RCNN.ROI_SIZE = 7
_C.FAST_RCNN.SPATIAL_SCALE = 1.0 / _C.RPN.ANCHOR_CREATOR.FEATURE_STRIDE
_C.FAST_RCNN.ROI_SIGMMA = 1.0

# ----------------------- FASTER_RCNN------------------------------------------------------#
_C.FASTER_RCNN = ConfigNode()
_C.FASTER_RCNN.FEATRUE_EXTRACTOR = 'pretrained_vgg16'
_C.FASTER_RCNN.NMS_THRESHOLD = 0.3
_C.FASTER_RCNN.SCORE_THRESHOLD = 0.7
_C.FASTER_RCNN.OFFSET_NORM_MEAN = [0.0, 0.0, 0.0, 0.0]
_C.FASTER_RCNN.OFFSET_NORM_STD = [0.1, 0.1, 0.2, 0.2]
_C.FASTER_RCNN.NUM_WORKERS = 2

_C.FASTER_RCNN.BATCH_SIZE = 1
_C.FASTER_RCNN.EPOCHS = 1000
_C.FASTER_RCNN.WEIGHT_DECAY = 0.0005
_C.FASTER_RCNN.LEARNING_RATE = 0.001
_C.FASTER_RCNN.STEP_SIZE = 25
_C.FASTER_RCNN.LEARNING_RATE_DECAY = 0.4
_C.FASTER_RCNN.MOMENTUM = 0.9

_C.FASTER_RCNN.RESUME = False
_C.FASTER_RCNN.PRETRAINED_MODEL_PATH = None
_C.FASTER_RCNN.CHECK_FREQUENCY = 100

# ----------------------- R-FCN ----------------------------------------#
_C.R_FCN = ConfigNode()
_C.R_FCN.POOL_SIZE = 7
_C.R_FCN.NUM_CLASSES = 20
_C.R_FCN.FEATURE_STRIDE = 16
_C.R_FCN.IN_CHANNELS = _C.RPN.FEATURE_CHANNELS

def get_default_config():
    """
    Get the default config.
    """
    return _C.clone()

def combine_configs(cfg_path1, cfg_path2=None):
    # Priority 3: get default configs
    cfg = get_default_config()    

    # Priority 2: merge from yaml config
    if cfg_path1 is not None and os.path.exists(cfg_path1):
        cfg.merge_from_file(cfg_path1)
    
    if cfg_path2 is not None and os.path.exists(cfg_path2):
        cfg.merge_from_file(cfg_path2)

    return cfg
