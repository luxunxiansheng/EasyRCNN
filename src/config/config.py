# #### BEGIN LICENSE BLOCK #####
# Version: MPL 1.1/GPL 2.0/LGPL 2.1
#
# The contents of this file are subject to the Mozilla Public License Version
# 1.1 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
# http://www.mozilla.org/MPL/
#
# Software distributed under the License is distributed on an "AS IS" basis,
# WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
# for the specific language governing rights and limitations under the
# License.
#
#
# Contributor(s):
#
#    Bin.Li (ornot2008@yahoo.com)
#
#
# Alternatively, the contents of this file may be used under the terms of
# either the GNU General Public License Version 2 or later (the "GPL"), or
# the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
# in which case the provisions of the GPL or the LGPL are applicable instead
# of those above. If you wish to allow use of your version of this file only
# under the terms of either the GPL or the LGPL, and not to allow others to
# use your version of this file under the terms of the MPL, indicate your
# decision by deleting the provisions above and replace them with the notice
# and other provisions required by the GPL or the LGPL. If you do not delete
# the provisions above, a recipient may use your version of this file under
# the terms of any one of the MPL, the GPL or the LGPL.
#
# #### END LICENSE BLOCK #####
#
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

# -------------------------- RPN ---------------------------------------------------#
_C.RPN = ConfigNode()

_C.RPN.BACKBONE = "vgg16"
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


# -----------------------RPN.TRAIN -----------------------------------------------------#
_C.RPN.TRAIN = ConfigNode()
_C.RPN.TRAIN.BATCH_SIZE = 1
_C.RPN.TRAIN.EPOCHS = 1000
_C.RPN.TRAIN.WEIGHT_DECAY = 0.0005
_C.RPN.TRAIN.LEARNING_RATE = 0.001
_C.RPN.TRAIN.MOMENTUM = 0.9
_C.RPN.TRAIN.LEARNING_RATE_DECAY = 0.1
_C.RPN.TRAIN.NUM_WORKERS = 8
_C.RPN.TRAIN.PRETRAINED_MODEL_PATH = None
_C.RPN.TRAIN.CHECK_FREQUENCY = 100

# ----------------------- FAST_RCNN -----------------------------------------------------#
_C.FAST_RCNN = ConfigNode()
_C.FAST_RCNN.IN_CHANNELS = _C.RPN.FEATURE_CHANNELS
_C.FAST_RCNN.FC7_CHANNELS = 1024
_C.FAST_RCNN.NUM_CLASSES = 20
_C.FAST_RCNN.ROI_SIZE = 7
_C.FAST_RCNN.SPATIAL_SCALE = 1.0 / _C.RPN.ANCHOR_CREATOR.FEATURE_STRIDE
_C.FAST_RCNN.ROI_SIGMMA = 1.0

# ----------------------- FASTER_RCNN------------------------------------------------------#
_C.FASTER_RCNN = ConfigNode()
_C.FASTER_RCNN.FEATRUE_EXTRACTOR = 'vgg16'
_C.FASTER_RCNN.EVALUATE_NMS_THRESHOLD = 0.3
_C.FASTER_RCNN.EVALUATE_SCORE_THRESHOLD = 0.05
_C.FASTER_RCNN.VISUAL_NMS_THRESHOLD = 0.3
_C.FASTER_RCNN.VISUAL_SCORE_THRESHOLD = 0.7
_C.FASTER_RCNN.OFFSET_NORM_MEAN = [0.0, 0.0, 0.0, 0.0]
_C.FASTER_RCNN.OFFSET_NORM_STD = [0.1, 0.1, 0.2, 0.2]

#-------------------------FASTER_RCNN.TRAIN-------------------------------------------------#
_C.FASTER_RCNN.TRAIN = ConfigNode()
_C.FASTER_RCNN.TRAIN.BATCH_SIZE = 1
_C.FASTER_RCNN.TRAIN.EPOCHS = 1000
_C.FASTER_RCNN.TRAIN.WEIGHT_DECAY = 0.0005
_C.FASTER_RCNN.TRAIN.LEARNING_RATE = 0.001
_C.FASTER_RCNN.TRAIN.MOMENTUM = 0.9
_C.FASTER_RCNN.TRAIN.LEARNING_RATE_DECAY = 0.1
_C.FASTER_RCNN.TRAIN.NUM_WORKERS = 2
_C.FASTER_RCNN.TRAIN.RESUME = False
_C.FASTER_RCNN.TRAIN.PRETRAINED_MODEL_PATH = None
_C.FASTER_RCNN.TRAIN.CHECK_FREQUENCY = 100
_C.FASTER_RCNN.TRAIN.RESUME = False
_C.FASTER_RCNN.TRAIN.RPN_CLS_LOSS_WEIGHT = 1.0
_C.FASTER_RCNN.TRAIN.RPN_REG_LOSS_WEIGHT = 1.0
_C.FASTER_RCNN.TRAIN.ROI_CLS_LOSS_WEIGHT = 1.0
_C.FASTER_RCNN.TRAIN.ROI_REG_LOSS_WEIGHT = 1.0


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
