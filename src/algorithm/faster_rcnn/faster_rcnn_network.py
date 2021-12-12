import torch
from torch import nn

from fast_rcnn.fast_rcnn_network import FastRCNN
from feature_extractor import FeatureExtractorFactory
from rpn.anchor_creator import AnchorCreator
from rpn.proposal_creator import ProposalCreator
from rpn.region_proposal_network import RPN
from utility import Utility

class FasterRCNN(nn.Module):
    def __init__(self,config,):
        super().__init__()
        self.feature_extractor = FeatureExtractorFactory.create_feature_extractor(config.FASTER_RCNN.FEATRUE_EXTRACTOR)
        self.rpn = RPN(config)
        self.fast_rcnn = FastRCNN(config)
        self.anchor_creator = AnchorCreator(config)
        self.proposal_creator = ProposalCreator(config)
        
    def forward(self,img):
        feature= self.feature_extractor(img)
        feature_height,feature_width = feature.size()[-2:]
        predicted_locs, predicted_scores = self.rpn(feature)

        anchors_of_img = self.anchor_creator.generate(feature_height,feature_width)
        img_height,img_width = img.shape[-2:]
        proposed_roi_bboxes =self.proposal_creator.generate(anchors_of_img,predicted_locs[0],predicted_scores[0],feature_height,feature_width,img_height,img_width)
        proposed_roi_bbox_indices = torch.zeros(len(proposed_roi_bboxes))
        predicted_roi_cls_loc,predicted_roi_cls_score = self.fast_rcnn(feature,proposed_roi_bboxes,proposed_roi_bbox_indices)
        predicted_roi_bboxes = Utility.loc_to_bbox(proposed_roi_bboxes,predicted_roi_cls_loc)
        return predicted_roi_cls_score,predicted_roi_bboxes
