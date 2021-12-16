"""
Default configurations for Faster R-CNN.
"""
import os

from yacs.config import CfgNode as ConfigNode

# -----------------------ROOT ----------------------------------------------------- #
_C = ConfigNode()

# ----------------------- TEST ----------------------------------------------------#
_C.TEST = ConfigNode()
_C.TEST.TEMP_DIR = '/media/yan/D/ornot/workspace/object_detection/test/temp'

# ----------------------- CHECKPOINT-----------------------------------------------#
_C.CHECKPOINT = ConfigNode()
_C.CHECKPOINT.CHECKPOINT_PATH = '/media/yan/D/ornot/workspace/object_detection/checkpoint/model_checkpoint.ckpt'

# ----------------------- DATASET ------------------------------------------------- #
_C.VOC_DATASET = ConfigNode()
_C.VOC_DATASET.DATA_DIR ='/home/yan/data/VOCdevkit/VOC2007'
_C.VOC_DATASET.USE_DIFFICULT_LABEL = False

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
_C.RPN.PROPOSAL_CREATOR.N_TRAIN_PRE_NMS = 12000
_C.RPN.PROPOSAL_CREATOR.N_TRAIN_POST_NMS = 2000
_C.RPN.PROPOSAL_CREATOR.N_TEST_PRE_NMS = 6000
_C.RPN.PROPOSAL_CREATOR.N_TEST_POST_NMS = 300
_C.RPN.PROPOSAL_CREATOR.MIN_SIZE = 16

# -----------------------RPN.PROPOAL_TARGET_CREATOR---------------#
_C.RPN.PROPOSAL_TARGET_CREATOR = ConfigNode()
_C.RPN.PROPOSAL_TARGET_CREATOR.N_SAMPLES = 128
_C.RPN.PROPOSAL_TARGET_CREATOR.POSITIVE_RATIO = 0.25
_C.RPN.PROPOSAL_TARGET_CREATOR.POSITIVE_IOU_THRESHOLD = 0.5
_C.RPN.PROPOSAL_TARGET_CREATOR.NEGATIVE_IOU_THRESHOLD_HI = 0.5
_C.RPN.PROPOSAL_TARGET_CREATOR.NEGATIVE_IOU_THRESHOLD_LO = 0.0

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

#-------------------------FASTER_RCNN.TRAIN-------------------------------------------------#
_C.FASTER_RCNN.TRAIN = ConfigNode()
_C.FASTER_RCNN.TRAIN.BATCH_SIZE = 1
_C.FASTER_RCNN.TRAIN.EPOCHS = 1000
_C.FASTER_RCNN.TRAIN.WEIGHT_DECAY = 0.0005
_C.FASTER_RCNN.TRAIN.LEARNING_RATE = 0.001
_C.FASTER_RCNN.TRAIN.MOMENTUM = 0.9
_C.FASTER_RCNN.TRAIN.LEARNING_RATE_DECAY = 0.1
_C.FASTER_RCNN.TRAIN.NUM_WORKERS = 8
_C.FASTER_RCNN.TRAIN.RESUME = False
_C.FASTER_RCNN.TRAIN.PRETRAINED_MODEL_PATH = None
_C.FASTER_RCNN.TRAIN.CHECK_FREQUENCY = 100
_C.FASTER_RCNN.TRAIN.RESUME = False


def get_default_config():
    """
    Get the default config.
    """
    return _C.clone()

def combine_configs(cfg_path):
    # Priority 3: get default configs
    cfg = get_default_config()    

    # Priority 2: merge from yaml config
    if cfg_path is not None and os.path.exists(cfg_path):
        cfg.merge_from_file(cfg_path)

    return cfg
